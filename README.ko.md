# Smart Connections MCP — v2 Hybrid

**Obsidian Vault에 실제로 존재하는 note를 빠르고 local하게 검색합니다.**

이 fork는 upstream v2의 모듈형 MCP server 위에 Smart Connections embedding,
disk 직접 발견, 다국어 BM25, EmbeddingGemma, 조건부 reranker를 결합합니다.
Vault를 cloud search service로 보내지 않으면서 장기 실행 agent가 의미 기반
recall과 대화형 속도를 함께 확보하는 것이 목적입니다.

[English](README.md) · [Upstream v2](https://github.com/msdanyg/smart-connections-mcp)

## 무엇이 달라졌나

- Plugin index에 아직 없는 새 Markdown도 disk에서 찾아 BM25로 검색합니다.
- 삭제·이동되어 disk에 없는 note는 결과에서 제거합니다.
- 기존 `.smart-env/embedding-index.json` v4 EmbeddingGemma index를 재사용합니다.
- Query마다 모든 고비용 단계를 실행하지 않고, `adaptive`가 빠른 검색 결과의
  합의를 먼저 확인한 뒤 필요할 때만 EmbeddingGemma와 작은 batched rerank를
  실행합니다.
- 각 결과가 어떤 retrieval leg에서 왔는지 `retrieval`과 `scoreType`으로
  표시합니다.
- `get_similar_notes`와 connection graph도 저장된 EmbeddingGemma note vector를
  우선 사용하여 plugin 미색인 note를 연결하고, 해당 vector가 없으면 plugin
  vector로 fallback합니다.

## Profile

| profile | 동작 | 권장 용도 |
|---|---|---|
| `adaptive` (기본) | fast 합의 후 필요할 때만 EmbeddingGemma + 6개 rerank | 대화형 agent |
| `fast` | Smart Connections dense + BM25 | 최저 지연 |
| `balanced` | dense + EmbeddingGemma + BM25 | top-10 recall 우선 |
| `quality` | 전체 retrieval + 큰 rerank pool | offline 정밀 검색 |
| `plugin` | upstream v2 검색 | 호환성·ablation |

## 설치

Node.js 20+, Smart Connections embedding이 생성된 Obsidian Vault, MCP client가
필요합니다.

```bash
git clone https://github.com/Hahyun-Lee/smart-connections-mcp.git
cd smart-connections-mcp
npm ci
npm run build
```

MCP 설정 예시:

```json
{
  "mcpServers": {
    "smart-connections": {
      "command": "node",
      "args": ["/absolute/path/to/smart-connections-mcp/dist/index.js"],
      "env": {
        "SMART_VAULT_PATH": "/absolute/path/to/Obsidian Vault",
        "SMART_SEARCH_PROFILE": "adaptive"
      }
    }
  }
}
```

현재 npm의 `smart-connections-mcp` package는 upstream release입니다. Hybrid
profile을 사용하려면 별도 package가 공개되기 전까지 위 source 설치를
사용하세요.

EmbeddingGemma index를 만들거나 증분 갱신하려면:

```bash
SMART_VAULT_PATH="/absolute/path/to/vault" npm run reindex
```

## 실사용 Vault A/B 결과

실제 사용 중인 804-note Vault에서 영문 의미, 한국어 의미, exact/common,
disk-only coverage를 포함한 65개 query로 측정했습니다.

| 구현/profile | R@1 | R@10 | median | p95 | stale top-10 |
|---|---:|---:|---:|---:|---:|
| 기존 v1 fork | 81.5% | 95.4% | 8,141 ms | 13,537 ms | 4 |
| v2 `adaptive` | 76.9% | 93.8% | 743 ms | 1,221 ms | 0 |
| v2 `balanced` | 56.9% | 95.4% | 92 ms | 227 ms | 0 |
| v2 `quality` | 67.7% | 98.5% | 3,706 ms | 4,173 ms | 0 |

`adaptive`는 기존보다 R@1 4.6%p, R@10 1.6%p 낮지만 median과 p95가 약
11배 빨라 운영 기본값으로 선택했습니다. 이는 단일 private Vault에 대한 운영
결정 근거이며 범용 benchmark 주장이 아닙니다. Private note 내용과 path는
공개하지 않습니다.

이 프로젝트는 Daniel Glickman의 MIT-licensed
[Smart Connections MCP](https://github.com/msdanyg/smart-connections-mcp)를 기반으로
합니다.
