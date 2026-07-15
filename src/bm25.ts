/** In-memory multilingual BM25 used as the lexical leg of hybrid retrieval. */

const STOP = new Set(['연구', '분석', '결과', 'the', 'a', 'an', 'of', 'and', 'to', 'in', 'is', 'for', 'with', 'on']);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .split(/[^a-z0-9가-힣]+/)
    .filter((token) => token.length > 0 && !STOP.has(token));
}

export interface BM25Document {
  id: string;
  text: string;
}

export class BM25Index {
  private tokens: string[][] = [];
  private ids: string[] = [];
  private df = new Map<string, number>();
  private avgLength = 0;

  build(documents: BM25Document[]): void {
    this.ids = documents.map((document) => document.id);
    this.tokens = documents.map((document) => tokenize(document.text));
    this.df.clear();
    for (const document of this.tokens) {
      for (const token of new Set(document)) this.df.set(token, (this.df.get(token) ?? 0) + 1);
    }
    const total = this.tokens.reduce((sum, document) => sum + document.length, 0);
    this.avgLength = total / (this.tokens.length || 1);
  }

  get size(): number {
    return this.ids.length;
  }

  topK(query: string, limit: number): Array<{ id: string; score: number }> {
    const queryTokens = tokenize(query);
    const count = this.tokens.length;
    const k1 = 1.5;
    const b = 0.75;
    const scores: Array<{ id: string; score: number }> = [];
    for (let index = 0; index < count; index++) {
      const document = this.tokens[index];
      const tf = new Map<string, number>();
      for (const token of document) tf.set(token, (tf.get(token) ?? 0) + 1);
      let score = 0;
      for (const token of queryTokens) {
        const frequency = this.df.get(token) ?? 0;
        if (!frequency) continue;
        const idf = Math.log((count - frequency + 0.5) / (frequency + 0.5) + 1);
        const occurrences = tf.get(token) ?? 0;
        score +=
          idf *
          (occurrences * (k1 + 1)) /
          (occurrences + k1 * (1 - b + (b * document.length) / this.avgLength));
      }
      if (score > 0) scores.push({ id: this.ids[index], score });
    }
    return scores.sort((a, b) => b.score - a.score).slice(0, limit);
  }
}
