/** Orchestrates search, similarity, graphs, content, and stats across vaults. */

import type { EmbedFn } from './embedder.js';
import { BM25Index } from './bm25.js';
import { BlockNotFoundError, EmbedUnavailableError } from './errors.js';
import { rerank } from './reranker.js';
import type {
  ConnectionGraph,
  SearchResponse,
  SearchResult,
  SimilarNote,
  VaultInfo,
} from './types.js';
import type { Vault } from './vault.js';
import type { VaultRegistry } from './vault-registry.js';
import type { IndexEntry } from './vector-index.js';

export interface QueryEmbedder {
  getEmbedFn(
    modelKey: string,
    parity?: { text: string; vec: number[] },
    warn?: (msg: string) => void,
  ): Promise<EmbedFn>;
}

const SNIPPET_MAX = 700;
const round = (n: number) => Math.round(n * 10_000) / 10_000;
export type SearchProfile = 'plugin' | 'fast' | 'balanced' | 'adaptive' | 'quality';

function profileFrom(value: string | undefined): SearchProfile {
  return value === 'plugin' || value === 'fast' || value === 'quality' || value === 'balanced' || value === 'adaptive'
    ? value
    : 'adaptive';
}

function likelyExactQuery(query: string): boolean {
  return /\b(?:19|20)\d{2}\b|\b10\.\d{4,9}\/\S+|["“”][^"“”]+["“”]/i.test(query);
}

export class SearchEngine {
  readonly profile: SearchProfile;
  private bm25 = new Map<string, { revision: number; index: BM25Index }>();

  constructor(
    private registry: VaultRegistry,
    private embedder: QueryEmbedder,
    options: { profile?: SearchProfile } = {},
  ) {
    this.profile = options.profile ?? profileFrom(process.env.SMART_SEARCH_PROFILE);
  }

  async search(
    query: string,
    opts: { vault?: string; scope?: 'notes' | 'blocks' | 'both'; limit?: number; threshold?: number } = {},
  ): Promise<SearchResponse> {
    if (this.profile === 'plugin' || opts.scope === 'blocks') {
      return this.pluginSearch(query, opts);
    }

    const { vault, scope = 'both', limit = 10, threshold = 0.4 } = opts;
    const vaults = this.registry.byName(vault);
    const poolSize = Math.max(20, limit * 3);
    const warnings: string[] = [];
    const fallbackVaults: string[] = [];
    const candidates: SearchResult[] = [];
    let semanticSucceeded = false;
    const exactQuery = likelyExactQuery(query);

    for (const v of vaults) {
      v.maybeReload();
      const rankedLists: Array<{
        name: 'plugin-dense' | 'embedding-gemma' | 'bm25';
        weight: number;
        items: Array<{ path: string; result: SearchResult }>;
      }> = [];

      try {
        const embed = await this.embedder.getEmbedFn(v.modelKey, v.paritySample(), (message) =>
          warnings.push(`${v.name}: ${message}`),
        );
        const vector = await embed(query);
        const filter = scope === 'notes' ? (entry: IndexEntry) => entry.kind === 'note' : undefined;
        const matches = v.index.topK(vector, poolSize, threshold, filter);
        rankedLists.push({
          name: 'plugin-dense',
          weight: 0.8,
          items: matches.map((match) => ({
            path: match.entry.notePath,
            result: this.toResult(v, match.entry, match.similarity),
          })),
        });
        semanticSucceeded = true;
      } catch (error) {
        if (!(error instanceof EmbedUnavailableError)) throw error;
        fallbackVaults.push(v.name);
      }

      const lexical = this.bm25For(v).topK(query, poolSize);
      rankedLists.push({
        name: 'bm25',
        weight: 0.8,
        items: lexical.map((hit) => ({
          path: hit.id,
          result: {
            path: hit.id,
            vault: v.name,
            similarity: hit.score,
            scope: 'note',
            snippet: v.noteSnippet(hit.id),
            match: 'keyword',
          },
        })),
      });
      const pluginTop = new Set(
        (rankedLists.find((list) => list.name === 'plugin-dense')?.items ?? [])
          .slice(0, 5)
          .map((item) => item.path),
      );
      const fastAgreement = lexical.slice(0, 5).some((item) => pluginTop.has(item.id));
      const adaptiveNeedsQuality = !exactQuery || !fastAgreement;
      const useGemma =
        (this.profile === 'balanced' || this.profile === 'adaptive' || this.profile === 'quality') &&
        !(this.profile === 'adaptive' && !adaptiveNeedsQuality);
      if (useGemma && v.gemma.hasEntries) {
        try {
          const hits = await v.gemma.search(query, v.notePathSet(), poolSize, 0.15);
          rankedLists.push({
            name: 'embedding-gemma',
            weight: 1,
            items: hits.map((hit) => ({
              path: hit.path,
              result: {
                path: hit.path,
                vault: v.name,
                similarity: hit.score,
                scope: 'note',
                snippet: v.noteSnippet(hit.path),
              },
            })),
          });
          semanticSucceeded = semanticSucceeded || hits.length > 0;
        } catch (error) {
          warnings.push(`${v.name}: EmbeddingGemma unavailable — ${error instanceof Error ? error.message : String(error)}`);
        }
      }

      const fused = new Map<string, { result: SearchResult; score: number; retrieval: string[]; semantic: boolean }>();
      for (const list of rankedLists) {
        const seen = new Set<string>();
        for (let rank = 0; rank < list.items.length; rank++) {
          const item = list.items[rank];
          if (seen.has(item.path)) continue;
          seen.add(item.path);
          const existing = fused.get(item.path) ?? {
            result: item.result,
            score: 0,
            retrieval: [],
            semantic: false,
          };
          existing.score += list.weight / (60 + rank + 1);
          if (!existing.retrieval.includes(list.name)) existing.retrieval.push(list.name);
          if (list.name !== 'bm25') {
            existing.semantic = true;
            if (item.result.scope === 'block' || existing.result.match === 'keyword') existing.result = item.result;
          }
          fused.set(item.path, existing);
        }
      }
      const ordered = [...fused.values()].sort((a, b) => b.score - a.score);
      const best = ordered[0]?.score || 1;
      for (const item of ordered.slice(0, poolSize)) {
        candidates.push({
          ...item.result,
          similarity: round(item.score / best),
          ...(item.semantic ? { match: undefined } : { match: 'keyword' as const }),
          retrieval: item.retrieval,
          scoreType: 'rrf',
        });
      }
    }

    let ranked = candidates.sort((a, b) => {
      const semanticOrder = Number(a.match === 'keyword') - Number(b.match === 'keyword');
      return semanticOrder || b.similarity - a.similarity;
    });
    const adaptiveNeedsReranker =
      this.profile === 'adaptive' &&
      ranked.some((item) => item.retrieval?.includes('embedding-gemma'));
    const useReranker = this.profile === 'quality' || adaptiveNeedsReranker;
    if (useReranker && ranked.length > 0) {
      try {
        const requestedPool = this.profile === 'adaptive'
          ? Number(process.env.SMART_RERANK_CANDIDATES ?? 6)
          : Math.max(limit * 2, 12);
        const rerankPool = ranked.slice(0, Math.max(1, Math.min(40, requestedPool)));
        const byId = new Map(rerankPool.map((item) => [`${item.vault}\u0000${item.path}`, item]));
        const reranked = await rerank(
          query,
          rerankPool.map((item) => ({
            id: `${item.vault}\u0000${item.path}`,
            text: `${item.path.replace(/\.md$/i, '').replace(/[-_]/g, ' ')}. ${item.snippet}`.slice(0, 2400),
          })),
        );
        const rerankedItems: SearchResult[] = reranked.map(({ id, score }) => {
            const item = byId.get(id)!;
            return {
              ...item,
              similarity: round(1 / (1 + Math.exp(-score))),
              retrieval: [...(item.retrieval ?? []), 'reranker'],
              scoreType: 'reranker' as const,
            };
          });
        ranked = [...rerankedItems, ...ranked.slice(rerankPool.length)];
      } catch (error) {
        warnings.push(`reranker unavailable — ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    if (fallbackVaults.length > 0) {
      warnings.push(
        `plugin embedding unavailable for ${fallbackVaults.join(', ')}; hybrid lexical/EmbeddingGemma paths remained available`,
      );
    }
    return {
      mode: semanticSucceeded ? 'semantic' : 'keyword-fallback',
      profile: this.profile,
      ...(warnings.length ? { warning: warnings.join(' | ') } : {}),
      results: ranked.slice(0, limit),
    };
  }

  private bm25For(vault: Vault): BM25Index {
    const cached = this.bm25.get(vault.name);
    if (cached?.revision === vault.revision) return cached.index;
    const index = new BM25Index();
    const documents = vault.notePaths().flatMap((notePath) => {
      try {
        const title = notePath.split('/').pop()!.replace(/\.md$/i, '').replace(/[-_]/g, ' ');
        return [{ id: notePath, text: `${title} ${vault.readNote(notePath).slice(0, 20_000)}` }];
      } catch {
        return [];
      }
    });
    index.build(documents);
    this.bm25.set(vault.name, { revision: vault.revision, index });
    return index;
  }

  private async pluginSearch(
    query: string,
    opts: { vault?: string; scope?: 'notes' | 'blocks' | 'both'; limit?: number; threshold?: number } = {},
  ): Promise<SearchResponse> {
    const { vault, scope = 'both', limit = 10, threshold = 0.4 } = opts;
    const vaults = this.registry.byName(vault);
    const warnings: string[] = [];
    const fallbackVaults: string[] = [];
    const results: SearchResult[] = [];

    const scopeFilter =
      scope === 'both' ? undefined : (e: IndexEntry) => e.kind === (scope === 'notes' ? 'note' : 'block');

    for (const v of vaults) {
      v.maybeReload();
      try {
        const embed = await this.embedder.getEmbedFn(v.modelKey, v.paritySample(), (m) =>
          warnings.push(`${v.name}: ${m}`),
        );
        const qvec = await embed(query);
        for (const m of v.index.topK(qvec, limit, threshold, scopeFilter)) {
          results.push(this.toResult(v, m.entry, m.similarity));
        }
      } catch (e) {
        if (!(e instanceof EmbedUnavailableError)) throw e;
        fallbackVaults.push(v.name);
        results.push(...this.keywordSearch(v, query, limit));
      }
    }

    const semanticRows = results.filter((r) => r.match === undefined).sort((a, b) => b.similarity - a.similarity);
    const fallbackRows = results.filter((r) => r.match === 'keyword').sort((a, b) => b.similarity - a.similarity);
    const ranked = [...semanticRows, ...fallbackRows];
    const allFellBack = fallbackVaults.length === vaults.length && vaults.length > 0;
    if (fallbackVaults.length > 0) {
      warnings.push(
        `semantic model unavailable for ${fallbackVaults.join(', ')} — used literal keyword matching there ` +
          `(scores are match counts, not cosine similarity)`,
      );
    }
    return {
      mode: allFellBack ? 'keyword-fallback' : 'semantic',
      profile: 'plugin',
      ...(warnings.length ? { warning: warnings.join(' | ') } : {}),
      results: ranked.slice(0, limit),
    };
  }

  private toResult(v: Vault, entry: IndexEntry, similarity: number): SearchResult {
    if (entry.kind === 'block') {
      return {
        path: entry.notePath,
        vault: v.name,
        similarity: round(similarity),
        scope: 'block',
        block: entry.id.slice(entry.notePath.length),
        snippet: v.blockSnippet(entry.id),
      };
    }
    return {
      path: entry.notePath,
      vault: v.name,
      similarity: round(similarity),
      scope: 'note',
      snippet: v.noteSnippet(entry.notePath),
    };
  }

  /** Literal keyword scoring — no RegExp built from user input, ever. */
  private keywordSearch(v: Vault, query: string, limit: number): SearchResult[] {
    const tokens = query.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
    if (tokens.length === 0) return [];
    const out: SearchResult[] = [];
    for (const notePath of v.data.sources.keys()) {
      let raw: string;
      try {
        raw = v.readNote(notePath);
      } catch {
        continue;
      }
      const content = raw.toLowerCase();
      let total = 0;
      let firstIdx = -1;
      for (const t of tokens) {
        let idx = content.indexOf(t);
        if (idx !== -1 && (firstIdx === -1 || idx < firstIdx)) firstIdx = idx;
        while (idx !== -1) {
          total++;
          idx = content.indexOf(t, idx + t.length);
        }
      }
      if (total === 0) continue;
      const start = Math.max(0, firstIdx - 200);
      out.push({
        path: notePath,
        vault: v.name,
        similarity: round(Math.min(total / 10, 1)),
        scope: 'note',
        snippet: raw.slice(start, start + SNIPPET_MAX),
        match: 'keyword',
      });
    }
    return out.sort((a, b) => b.similarity - a.similarity).slice(0, limit);
  }

  getSimilarNotes(
    notePath: string,
    opts: { vault?: string; threshold?: number; limit?: number } = {},
  ): SimilarNote[] {
    const { vault, threshold = 0.5, limit = 10 } = opts;
    for (const candidate of this.registry.byName(vault)) candidate.maybeReload();
    const v = this.registry.resolveNote(notePath, vault);
    const gemmaSimilar = v.gemma.similarByPath(notePath, v.notePathSet(), limit, threshold);
    if (gemmaSimilar) {
      return gemmaSimilar.map((item) => ({
        path: item.path,
        vault: v.name,
        similarity: round(item.score),
        blocks: Object.keys(v.data.sources.get(item.path)?.blocks ?? {}),
      }));
    }
    const vec = v.data.sources.get(notePath)?.embeddings?.[v.modelKey]?.vec;
    if (!vec) throw new EmbedUnavailableError(`No stored embedding for note: ${notePath}`);
    return v.index
      .topK(vec, limit, threshold, (e) => e.kind === 'note' && e.notePath !== notePath)
      .map((m) => ({
        path: m.entry.notePath,
        vault: v.name,
        similarity: round(m.similarity),
        blocks: Object.keys(v.data.sources.get(m.entry.notePath)?.blocks ?? {}),
      }));
  }

  getConnectionGraph(
    notePath: string,
    opts: { vault?: string; depth?: number; threshold?: number; maxPerLevel?: number } = {},
  ): ConnectionGraph {
    const { vault, depth = 2, threshold = 0.6, maxPerLevel = 5 } = opts;
    for (const candidate of this.registry.byName(vault)) candidate.maybeReload();
    const v = this.registry.resolveNote(notePath, vault);
    const visited = new Set<string>();
    const connections: ConnectionGraph['connections'] = [];

    const walk = (current: string, level: number, similarity: number): void => {
      visited.add(current);
      if (level > 0) connections.push({ path: current, depth: level, similarity: round(similarity) });
      if (level >= depth) return;
      let similar: SimilarNote[];
      try {
        similar = this.getSimilarNotes(current, { vault: v.name, threshold, limit: maxPerLevel });
      } catch (e) {
        // A node beyond the root may lack a stored embedding — skip expanding it.
        if (level > 0 && e instanceof EmbedUnavailableError) return;
        throw e;
      }
      for (const s of similar) {
        if (!visited.has(s.path)) walk(s.path, level + 1, s.similarity);
      }
    };

    walk(notePath, 0, 1);
    return { root: notePath, vault: v.name, connections };
  }

  getNoteContent(
    notePath: string,
    opts: { vault?: string; includeBlocks?: string[] } = {},
  ): object {
    for (const candidate of this.registry.byName(opts.vault)) candidate.maybeReload();
    const v = this.registry.resolveNote(notePath, opts.vault);
    const blocks = Object.keys(v.data.sources.get(notePath)?.blocks ?? {});
    if (opts.includeBlocks && opts.includeBlocks.length > 0) {
      const extracted: Record<string, string> = {};
      const missing: string[] = [];
      for (const heading of opts.includeBlocks) {
        try {
          extracted[heading] = v.extractBlockByHeading(notePath, heading);
        } catch (e) {
          if (e instanceof BlockNotFoundError) {
            missing.push(heading);
            continue;
          }
          throw e;
        }
      }
      return { path: notePath, vault: v.name, blocks, extracted, missing };
    }
    return { path: notePath, vault: v.name, blocks, content: v.readNote(notePath) };
  }

  listVaults(): VaultInfo[] {
    const ok: VaultInfo[] = this.registry.vaults.map((v) => ({
      name: v.name,
      path: v.path,
      status: 'ok' as const,
      ...v.stats(),
    }));
    const failed: VaultInfo[] = this.registry.failures.map((f) => ({
      name: f.name,
      path: f.path,
      status: 'error' as const,
      error: f.error,
    }));
    return [...ok, ...failed];
  }

  getStats(vaultName?: string): object {
    for (const candidate of this.registry.byName(vaultName)) candidate.maybeReload();
    const vaults = this.registry.byName(vaultName);
    const perVault = vaults.map((v) => ({ name: v.name, ...v.stats() }));
    return {
      vaults: perVault,
      totals: {
        notes: perVault.reduce((sum, s) => sum + s.notes, 0),
        blocks: perVault.reduce((sum, s) => sum + s.blocks, 0),
        indexed: perVault.reduce((sum, s) => sum + s.indexed, 0),
      },
    };
  }
}
