# Smart Connections MCP — v2 Hybrid

**Fast, local retrieval over the notes that actually exist in your Obsidian vault.**

This fork keeps the modular v2 MCP server and adds adaptive hybrid retrieval:
Smart Connections embeddings, direct disk discovery, multilingual BM25,
EmbeddingGemma, and an optional reranker. It is intended for long-running agents
that need both semantic recall and predictable interactive latency without
sending the vault to a cloud search service.

[한국어](README.ko.md) · [Upstream v2](https://github.com/msdanyg/smart-connections-mcp)

## Why this fork exists

A plugin embedding index alone can miss new files, return deleted paths, or spend
too long reranking every candidate. A lexical-only search misses paraphrases.
This server treats those as separate retrieval legs and chooses how much work to
do per query.

- `search_notes` searches whole notes or plugin-indexed blocks and returns a
  short snippet, the contributing retrieval legs, and the score type.
- Markdown is reconciled from disk, so plugin-unindexed notes remain searchable
  and deleted notes are filtered out.
- EmbeddingGemma is an optional, independent semantic index. Existing v4
  `.smart-env/embedding-index.json` files are reused.
- All inference and indexing are local. Model files are downloaded once and then
  cached by Transformers.js.
- The upstream tools remain: `get_similar_notes`, `get_connection_graph`,
  `get_note_content`, `get_stats`, and `list_vaults`.

## Search profiles

Set `SMART_SEARCH_PROFILE` to one of:

| profile | retrieval path | use it when |
|---|---|---|
| `adaptive` (default) | fast agreement first; conditional EmbeddingGemma + 6-item rerank | interactive agent use |
| `fast` | Smart Connections dense + BM25 | latency matters most |
| `balanced` | dense + EmbeddingGemma + BM25 | top-10 coverage matters more than first-result precision |
| `quality` | all legs + a larger batched rerank | offline/high-recall inspection |
| `plugin` | upstream v2 behavior | compatibility or ablation |

The adaptive rerank pool can be changed with `SMART_RERANK_CANDIDATES`.

## Install from this repository

Requires Node.js 20+, an Obsidian vault with Smart Connections embeddings, and
an MCP client such as Claude Desktop or Claude Code.

```bash
git clone https://github.com/Hahyun-Lee/smart-connections-mcp.git
cd smart-connections-mcp
npm ci
npm run build
```

Configure the MCP client with the built entry point:

```json
{
  "mcpServers": {
    "smart-connections": {
      "command": "node",
      "args": ["/absolute/path/to/smart-connections-mcp/dist/index.js"],
      "env": {
        "SMART_VAULT_PATH": "/absolute/path/to/Vault One,/absolute/path/to/Vault Two",
        "SMART_SEARCH_PROFILE": "adaptive"
      }
    }
  }
}
```

`SMART_VAULT_PATHS` is accepted as an alias and takes precedence when both forms
are set. Restart the MCP client after changing its configuration.

The npm package named `smart-connections-mcp` currently refers to the upstream
release, not this hybrid fork. Use the source installation above for the hybrid
profiles until a separately named package is published.

## Optional EmbeddingGemma index

`adaptive`, `balanced`, and `quality` can reuse an existing compatible index. To
create or incrementally refresh it:

```bash
SMART_VAULT_PATH="/absolute/path/to/vault" npm run reindex
```

The first run downloads `onnx-community/embeddinggemma-300m-ONNX`. If the index
is absent or the model cannot load, plugin-dense and BM25 retrieval remain
available and the response includes a warning.

## Operational benchmark

We compared the previous production fork and this migration on one actively used
804-note vault with 65 English semantic, Korean semantic, exact/common, and
disk-only coverage queries.

| implementation/profile | R@1 | R@10 | median | p95 | stale top-10 hits |
|---|---:|---:|---:|---:|---:|
| previous v1 fork | 81.5% | 95.4% | 8,141 ms | 13,537 ms | 4 |
| v2 `adaptive` | 76.9% | 93.8% | 743 ms | 1,221 ms | 0 |
| v2 `balanced` | 56.9% | 95.4% | 92 ms | 227 ms | 0 |
| v2 `quality` | 67.7% | 98.5% | 3,706 ms | 4,173 ms | 0 |

`adaptive` is the operational default because it is about 11× faster at median
and p95 while keeping most of the old ranking quality. The previous fork still
has higher R@1; this is a measured tradeoff, not a claim of universal
superiority. The dataset is one private vault, so these results support this
deployment decision rather than a general multi-domain benchmark claim. Private
note text and paths are not published.

## Migrating from upstream v2

The six tools and v2 response envelope remain. `search_notes` adds `profile`,
`retrieval`, and `scoreType`; clients that already read the v2 `results` array
continue to work. Use `SMART_SEARCH_PROFILE=plugin` for upstream v2 search
behavior.

## Development and verification

```bash
npm test
npm run test:live
npm run smoke -- "/path/to/vault" "your query"
```

This work is based on Daniel Glickman's MIT-licensed
[Smart Connections MCP](https://github.com/msdanyg/smart-connections-mcp).
See [CHANGELOG.md](CHANGELOG.md) and [LICENSE](LICENSE).
