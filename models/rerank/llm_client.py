from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI
from google import genai as google_genai
from google.genai import types as google_types


class LLMClient(ABC):
    @abstractmethod
    def chat_complete(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        user: Optional[str] = None,
    ) -> str: ...


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_complete(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        user: Optional[str] = None,
    ) -> str:
        completion = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            user=user,
        )
        return (completion.choices[0].message.content or "").strip()


class GeminiClient(LLMClient):
    def __init__(self, api_key: str):
        self._client = google_genai.Client(api_key=api_key)

    def chat_complete(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        user: Optional[str] = None,  # not supported by Gemini, ignored
    ) -> str:
        system_instruction: Optional[str] = None
        chat_messages = messages

        if chat_messages and chat_messages[0]["role"] == "system":
            system_instruction = chat_messages[0]["content"]
            chat_messages = chat_messages[1:]

        contents = [
            google_types.Content(
                role="model" if msg["role"] == "assistant" else "user",
                parts=[google_types.Part(text=msg["content"])],
            )
            for msg in chat_messages
        ]

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=google_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0,
                max_output_tokens=max_tokens,
            ),
        )
        return (response.text or "").strip()


def create_client(credentials: dict) -> LLMClient:
    provider = credentials.get("provider", "openai")
    if provider == "gemini":
        return GeminiClient(api_key=credentials["gemini_api_key"])
    return OpenAIClient(
        api_key=credentials.get("openai_api_key"),
        base_url=credentials.get("openai_base_url") or None,
    )
