/** Optional batched multilingual cross-encoder reranker. */

const MODEL_ID = 'onnx-community/bge-reranker-v2-m3-ONNX';

let tokenizer: any = null;
let model: any = null;
let loadPromise: Promise<void> | null = null;

async function ensureLoaded(): Promise<void> {
  if (tokenizer && model) return;
  if (!loadPromise) {
    loadPromise = (async () => {
      const { AutoModelForSequenceClassification, AutoTokenizer } = await import('@huggingface/transformers');
      console.error(`[reranker] loading ${MODEL_ID}`);
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
      model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, { dtype: 'q8' });
      console.error('[reranker] ready');
    })();
  }
  try {
    await loadPromise;
  } catch (error) {
    loadPromise = null;
    throw error;
  }
}

export interface RerankCandidate {
  id: string;
  text: string;
}

export async function rerank(
  query: string,
  candidates: RerankCandidate[],
): Promise<Array<{ id: string; score: number }>> {
  if (candidates.length === 0) return [];
  await ensureLoaded();
  const inputs = await tokenizer(Array(candidates.length).fill(query), {
    text_pair: candidates.map((candidate) => candidate.text),
    padding: true,
    truncation: true,
    max_length: 512,
  });
  const { logits } = await model(inputs);
  const raw = Array.from(logits.data as ArrayLike<number>);
  const stride = Math.max(1, Math.floor(raw.length / candidates.length));
  return candidates
    .map((candidate, index) => ({
      id: candidate.id,
      score: Number(raw[index * stride + (stride > 1 ? stride - 1 : 0)]),
    }))
    .sort((a, b) => b.score - a.score);
}
