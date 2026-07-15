/** Optional independent EmbeddingGemma index compatible with the production v4 file. */

import * as crypto from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { cosineSimilarity } from './vector-index.js';

const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX';
const DIMENSION = 768;
const INDEX_VERSION = 4;
const QUERY_PREFIX = 'task: search result | query: ';
const DOCUMENT_PREFIX = 'title: none | text: ';

interface GemmaEntry {
  vec: number[];
  note_path: string;
  block_key: string;
  block_type: string;
  char_length: number;
  hash: string;
  updated_at: number;
}

interface GemmaFile {
  model: string;
  dimension: number;
  version: number;
  created_at: number;
  updated_at: number;
  entries: Record<string, GemmaEntry>;
}

let tokenizer: any = null;
let model: any = null;
let loadPromise: Promise<void> | null = null;

function normalize(vector: number[]): number[] {
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return magnitude === 0 ? vector : vector.map((value) => value / magnitude);
}

async function ensureModel(): Promise<void> {
  if (tokenizer && model) return;
  if (!loadPromise) {
    loadPromise = (async () => {
      const { AutoModel, AutoTokenizer } = await import('@huggingface/transformers');
      console.error(`[embedding-gemma] loading ${MODEL_ID}`);
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
      model = await AutoModel.from_pretrained(MODEL_ID, { dtype: 'q8' });
      console.error('[embedding-gemma] ready');
    })();
  }
  try {
    await loadPromise;
  } catch (error) {
    loadPromise = null;
    throw error;
  }
}

async function embed(text: string, query: boolean): Promise<number[]> {
  await ensureModel();
  const prefix = query ? QUERY_PREFIX : DOCUMENT_PREFIX;
  const inputs = await tokenizer([prefix + text], {
    padding: true,
    truncation: true,
    max_length: 2048,
  });
  const outputs = await model(inputs);
  const tensor = outputs.sentence_embedding ?? outputs.last_hidden_state;
  if (!tensor?.data) throw new Error('EmbeddingGemma output has no sentence embedding');
  return normalize(Array.from(tensor.data as ArrayLike<number>).slice(0, DIMENSION));
}

function emptyIndex(): GemmaFile {
  const now = Date.now();
  return {
    model: MODEL_ID,
    dimension: DIMENSION,
    version: INDEX_VERSION,
    created_at: now,
    updated_at: now,
    entries: {},
  };
}

export class EmbeddingGemmaIndex {
  private readonly indexPath: string;
  private index: GemmaFile = emptyIndex();
  private loadError?: string;

  constructor(vaultPath: string) {
    this.indexPath = path.join(vaultPath, '.smart-env', 'embedding-index.json');
    this.load();
  }

  private load(): void {
    if (!fs.existsSync(this.indexPath)) return;
    try {
      const parsed = JSON.parse(fs.readFileSync(this.indexPath, 'utf-8')) as GemmaFile;
      if (parsed.version < INDEX_VERSION || parsed.model !== MODEL_ID || parsed.dimension !== DIMENSION) {
        throw new Error(`incompatible index v${parsed.version} model=${parsed.model}`);
      }
      this.index = parsed;
    } catch (error) {
      this.loadError = error instanceof Error ? error.message : String(error);
      this.index = emptyIndex();
    }
  }

  get hasEntries(): boolean {
    return Object.keys(this.index.entries).some((key) => this.index.entries[key].block_type === 'full');
  }

  searchStoredVector(
    vector: number[],
    allowedPaths: Set<string>,
    limit: number,
    threshold: number,
  ): Array<{ path: string; score: number }> {
    const results: Array<{ path: string; score: number }> = [];
    for (const entry of Object.values(this.index.entries)) {
      if (entry.block_type !== 'full' || !allowedPaths.has(entry.note_path)) continue;
      const score = cosineSimilarity(vector, entry.vec);
      if (score >= threshold) results.push({ path: entry.note_path, score });
    }
    return results.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  similarByPath(
    notePath: string,
    allowedPaths: Set<string>,
    limit: number,
    threshold: number,
  ): Array<{ path: string; score: number }> | null {
    const source = Object.values(this.index.entries).find(
      (entry) => entry.block_type === 'full' && entry.note_path === notePath,
    );
    if (!source) return null;
    return this.searchStoredVector(source.vec, allowedPaths, limit + 1, threshold)
      .filter((item) => item.path !== notePath)
      .slice(0, limit);
  }

  async search(
    query: string,
    allowedPaths: Set<string>,
    limit: number,
    threshold: number = 0.2,
  ): Promise<Array<{ path: string; score: number }>> {
    if (!this.hasEntries) return [];
    return this.searchStoredVector(await embed(query, true), allowedPaths, limit, threshold);
  }

  async sync(
    notePaths: string[],
    readNote: (notePath: string) => string,
  ): Promise<{ updated: number; unchanged: number; deleted: number; skipped: number }> {
    const current = new Set(notePaths);
    let deleted = 0;
    for (const [key, entry] of Object.entries(this.index.entries)) {
      if (!current.has(entry.note_path)) {
        delete this.index.entries[key];
        deleted++;
      }
    }
    let updated = 0;
    let unchanged = 0;
    let skipped = 0;
    for (const notePath of notePaths) {
      let content: string;
      try {
        content = readNote(notePath);
      } catch {
        skipped++;
        continue;
      }
      if (content.trim().length < 30) {
        skipped++;
        continue;
      }
      const hash = crypto.createHash('sha256').update(content).digest('hex');
      const previous = this.index.entries[notePath];
      if (previous?.block_type === 'full' && previous.hash === hash) {
        unchanged++;
        continue;
      }
      for (const [key, entry] of Object.entries(this.index.entries)) {
        if (entry.note_path === notePath) delete this.index.entries[key];
      }
      this.index.entries[notePath] = {
        vec: await embed(content, false),
        note_path: notePath,
        block_key: '__full__',
        block_type: 'full',
        char_length: content.length,
        hash,
        updated_at: Date.now(),
      };
      updated++;
    }
    this.index.updated_at = Date.now();
    const temporary = `${this.indexPath}.tmp-${process.pid}`;
    fs.writeFileSync(temporary, JSON.stringify(this.index), 'utf-8');
    fs.renameSync(temporary, this.indexPath);
    return { updated, unchanged, deleted, skipped };
  }

  stats(allowedPaths?: Set<string>): object {
    const fullEntries = Object.values(this.index.entries).filter((entry) => entry.block_type === 'full');
    const live = allowedPaths ? fullEntries.filter((entry) => allowedPaths.has(entry.note_path)).length : fullEntries.length;
    return {
      model: MODEL_ID,
      dimension: DIMENSION,
      notes: fullEntries.length,
      liveNotes: live,
      staleNotes: fullEntries.length - live,
      updatedAt: this.index.updated_at,
      ...(this.loadError ? { loadError: this.loadError } : {}),
    };
  }
}
