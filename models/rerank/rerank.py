from typing import Optional

import httpx
import re
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from dify_plugin import RerankModel
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import AIModelEntity, FetchFrom, ModelType
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.entities.model.rerank import RerankDocument, RerankResult

from .llm_client import LLMClient, create_client


class RankgptRerankModel(RerankModel):
    """
    Model class for Rankgpt rerank model.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        query: str,
        docs: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> RerankResult:
        if len(docs) == 0:
            return RerankResult(model=model, docs=[])

        if top_n is None:
            top_n = len(docs)

        window_size = int(credentials.get("window_size") or 0)
        step_size = int(credentials.get("step_size") or 0)
        max_doc_words = int(credentials.get("max_doc_words") or 300)

        client = create_client(credentials)

        ranked_indices = self._rank_documents_with_sliding_windows(
            client=client,
            model=model,
            query=query,
            docs=docs,
            window_size=window_size,
            step_size=step_size,
            max_doc_words=max_doc_words,
            user=user,
        )

        rerank_documents: list[RerankDocument] = []
        selected_indices = ranked_indices[:top_n]
        for final_rank, doc_idx in enumerate(selected_indices):
            # RankGPT returns permutation only (no true relevance score),
            # so we expose a rank-based pseudo-score for Dify.
            score = self._rank_to_score(final_rank=final_rank)
            rerank_document = RerankDocument(
                index=doc_idx,
                text=docs[doc_idx],
                score=score,
            )
            if score_threshold is not None and rerank_document.score < score_threshold:
                continue
            rerank_documents.append(rerank_document)

        return RerankResult(model=model, docs=rerank_documents)

    def _rank_documents_with_sliding_windows(
        self,
        client: LLMClient,
        model: str,
        query: str,
        docs: list[str],
        window_size: int,
        step_size: int,
        max_doc_words: int,
        user: Optional[str] = None,
    ) -> list[int]:
        ranked_indices = list(range(len(docs)))

        if window_size <= 0 or step_size <= 0 or len(docs) <= window_size:
            return self._rank_one_window(
                client=client,
                model=model,
                query=query,
                docs=docs,
                indices=ranked_indices,
                max_doc_words=max_doc_words,
                user=user,
            )

        rank_start = 0
        rank_end = len(docs)
        end_pos = rank_end
        start_pos = rank_end - window_size

        while start_pos >= rank_start:
            start_pos = max(start_pos, rank_start)
            current_window = ranked_indices[start_pos:end_pos]
            new_order = self._rank_one_window(
                client=client,
                model=model,
                query=query,
                docs=docs,
                indices=current_window,
                max_doc_words=max_doc_words,
                user=user,
            )
            ranked_indices[start_pos:end_pos] = new_order

            end_pos = end_pos - step_size
            start_pos = start_pos - step_size

        return ranked_indices

    def _rank_one_window(
        self,
        client: LLMClient,
        model: str,
        query: str,
        docs: list[str],
        indices: list[int],
        max_doc_words: int,
        user: Optional[str] = None,
    ) -> list[int]:
        num = len(indices)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are RankGPT, an intelligent assistant that ranks passages by "
                    "their relevance to a search query."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"I will provide {num} passages identified by numbers in brackets. "
                    f"Rank them by relevance to this query: {query}"
                ),
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

        for i, doc_idx in enumerate(indices, start=1):
            content = " ".join((docs[doc_idx] or "").split()[:max_doc_words]).strip()
            messages.append({"role": "user", "content": f"[{i}] {content}"})
            messages.append({"role": "assistant", "content": f"Received passage [{i}]."})

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Search Query: {query}\n"
                    f"Rank the {num} passages in descending relevance and output only ids "
                    "in this exact format: [1] > [2] > [3]."
                ),
            }
        )

        response = client.chat_complete(
            model=model,
            messages=messages,
            max_tokens=min(512, 8 * num + 16),
            user=user,
        )

        relative_order = self._parse_rank_response(response=response, total=num)
        return [indices[i] for i in relative_order]

    def _parse_rank_response(self, response: str, total: int) -> list[int]:
        candidates = [int(x) - 1 for x in re.findall(r"\d+", response)]
        seen = set()
        order: list[int] = []
        for idx in candidates:
            if 0 <= idx < total and idx not in seen:
                seen.add(idx)
                order.append(idx)

        for idx in range(total):
            if idx not in seen:
                order.append(idx)

        return order

    def _rank_to_score(self, final_rank: int) -> float:
        # Reciprocal-rank style score: 1.0, 0.5, 0.333..., ...
        return 1.0 / float(final_rank + 1)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            self._invoke(
                model=model,
                credentials=credentials,
                query="What is the capital of the United States?",
                docs=[
                    "Carson City is the capital city of the American state of Nevada. At the 2010 United States "
                    "Census, Carson City had a population of 55,274.",
                    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that "
                    "are a political division controlled by the United States. Its capital is Saipan.",
                ],
                score_threshold=0.8,
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [httpx.ConnectError, APIConnectionError, APITimeoutError],
            InvokeServerUnavailableError: [httpx.RemoteProtocolError, InternalServerError],
            InvokeRateLimitError: [RateLimitError],
            InvokeAuthorizationError: [httpx.HTTPStatusError, AuthenticationError, PermissionDeniedError],
            InvokeBadRequestError: [httpx.RequestError, BadRequestError, UnprocessableEntityError],
        }

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.RERANK,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={},
        )
