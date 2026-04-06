# RankGPT Dify Plugin

OpenAI-based RankGPT reranker plugin for Dify.

**Author:** [ki3dn](https://github.com/ki3nd)   
**Type:** model   
**Github Repo:** [https://github.com/ki3nd/rankgpt-dify-plugin](https://github.com/ki3nd/rankgpt-dify-plugin)   
**Github Issues:** [issues](https://github.com/ki3nd/rankgpt-dify-plugin/issues)  

## Overview

This plugin adds a `rerank model provider` to Dify and uses GPT to reorder retrieved documents by query relevance.

## Features

- OpenAI-only integration (`openai_api_key`, optional `openai_base_url`)
- RankGPT-style permutation reranking
- Sliding-window reranking for longer document lists
- Rank-based pseudo-score output (`1/(rank+1)`)

## Configure In Dify

When configuring the `rankgpt` provider in Dify:

- `model`: e.g. `gpt-4o-mini`
- `openai_api_key`: OpenAI API key
- `openai_base_url` (optional): for OpenAI-compatible endpoints
- `window_size` (optional, default `20`)
- `step_size` (optional, default `10`)
- `max_doc_words` (optional, default `300`)

<p align="center">
  <img src="./_assets/settings.png" alt="Dify Provider Settings" />
</p>

## How Reranking Works

1. The plugin builds a RankGPT-style prompt with the query and indexed passages.
2. GPT returns a ranking order (for example: `[3] > [1] > [2]`).
3. The plugin parses the order, removes duplicates and invalid items, and applies safe fallback for missing indices.
4. Results are returned to Dify as `RerankResult`.

## Notes

- This is LLM-based reranking, so latency and cost depend on model choice and document count.
- For large document sets, tune `window_size` and `step_size` to avoid oversized prompts.
- `score_threshold` is applied to a rank-based pseudo-score, not a true model relevance probability.
