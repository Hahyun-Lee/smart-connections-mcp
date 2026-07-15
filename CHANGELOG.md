# Changelog

## 2.0.0-brainai.1 — 2026-07-15

### Added
- Search profiles: upstream-compatible `plugin`, low-latency `fast`,
  high-recall `balanced`, conditional `adaptive`, and reranked `quality`.
- Optional EmbeddingGemma 300M index with an incremental `smart-connections-reindex`
  command and compatibility with the existing v4 index file.
- Multilingual BM25 over Markdown discovered directly from disk, including notes
  not yet represented in the Smart Connections plugin index.
- Batched bge-reranker-v2-m3 for the conditional and quality profiles.
- Retrieval provenance and score type on every hybrid result.
- EmbeddingGemma-first similar-note and connection-graph traversal, with plugin
  vector fallback for notes absent from the independent index.

### Fixed
- Cap the actual tokenizer configuration to the embedding model's position limit,
  preventing long-query ONNX failures without silently disabling semantic search.
- Filter deleted or moved notes against current disk state before returning them.

### Operational evidence
- On the actively used 804-note vault and 65-query internal set, `adaptive`
  measured 76.9% R@1 and 93.8% R@10 at 743 ms median. The previous v1 fork
  measured 81.5% / 95.4% at 8,141 ms. These are single-vault operational
  measurements, not a general benchmark claim.

## 2.0.0 — 2026-07-13

### Added
- **True semantic search**: `search_notes` now embeds your query locally with the
  same model your vault's Smart Connections index used (via transformers.js).
  Conceptual queries work; nothing leaves your machine.
- **Multi-vault**: `SMART_VAULT_PATH` accepts comma-separated paths; tools take an
  optional `vault` parameter; new `list_vaults` tool.
- **Block-level retrieval**: search matches individual sections (blocks) and
  returns content snippets inline.
- **Freshness**: `.smart-env` changes are picked up automatically (throttled
  incremental reload) — no server restart after editing notes.
- `get_note_content` `include_blocks` now actually extracts the named blocks.
- Explicit `mode: "keyword-fallback"` + warning when the embedding model cannot
  load, instead of silently degraded results.

### Fixed
- Path traversal in `get_note_content` (reads outside the vault are rejected).
- Crash on regex metacharacters in search queries.
- `.ajson` deletion entries (`null`) are now honored.

### Removed (breaking)
- `get_embedding_neighbors` tool.
- `search_notes` response shape changed (adds `vault`, `scope`, `block`,
  `snippet`, `mode`).

### Changed
- Requires Node >= 20. MCP SDK updated to the current 1.x line.
- First run downloads the embedding model (~25MB, cached locally forever).
