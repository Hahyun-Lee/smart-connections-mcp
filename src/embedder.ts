/**
 * Embeds query text locally with the same model a vault's Smart Connections data used.
 * Model weights download once (transformers.js cache) and run fully offline after that.
 */

import { EmbedUnavailableError } from './errors.js';
import { cosineSimilarity } from './vector-index.js';

export type EmbedFn = (text: string) => Promise<number[]>;

export type RawExtractor = (
  text: string,
  opts: { pooling: 'mean'; normalize: boolean },
) => Promise<{ data: ArrayLike<number> }>;

export type PipelineFactory = (
  modelId: string,
  opts: { dtype: 'fp32' | 'q8' },
) => Promise<RawExtractor>;

export function configureTokenizerLimit(
  tokenizer: {
    readonly model_max_length?: number;
    _tokenizerConfig?: { model_max_length?: number };
    config?: { model_max_length?: number };
  },
  modelConfig: { max_position_embeddings?: number } | undefined,
): number {
  const configured = Number(modelConfig?.max_position_embeddings);
  const modelLimit = Number.isFinite(configured) && configured > 0 ? Math.floor(configured) : 512;
  const advertised = Number(tokenizer.model_max_length);
  const safeLimit = Number.isFinite(advertised) && advertised > 0
    ? Math.min(Math.floor(advertised), modelLimit)
    : modelLimit;
  // transformers.js exposes model_max_length as a getter backed by
  // _tokenizerConfig, so assigning the public property throws in strict mode.
  if (tokenizer._tokenizerConfig) tokenizer._tokenizerConfig.model_max_length = safeLimit;
  if (tokenizer.config) tokenizer.config.model_max_length = safeLimit;
  return safeLimit;
}

const defaultFactory: PipelineFactory = async (modelId, opts) => {
  const { pipeline } = await import('@huggingface/transformers');
  const p = await pipeline('feature-extraction', modelId, { dtype: opts.dtype });
  // The feature-extraction pipeline requests truncation, but some tokenizers
  // advertise an unusably large model_max_length. ONNX then receives more
  // positions than the model supports (the v2.0.0 parity-probe failure). Bind
  // truncation to the model's real positional limit before any parity/query run.
  configureTokenizerLimit(
    (p as unknown as {
      tokenizer: {
        readonly model_max_length?: number;
        _tokenizerConfig?: { model_max_length?: number };
        config?: { model_max_length?: number };
      };
    }).tokenizer,
    (p as unknown as { model?: { config?: { max_position_embeddings?: number } } }).model?.config,
  );
  return p as unknown as RawExtractor;
};

const PARITY_WARN_THRESHOLD = 0.99;

// ~375 tokens of English prose — safely under the 512-token position limit of
// small embedding models whose tokenizers ship without a usable model_max_length
// (TaylorAI/bge-micro-v2 crashes onnxruntime on longer inputs).
const EMBED_MAX_CHARS = 1500;

export class Embedder {
  private factory: PipelineFactory;
  private cache = new Map<string, Promise<EmbedFn>>();

  constructor(factory: PipelineFactory = defaultFactory) {
    this.factory = factory;
  }

  getEmbedFn(
    modelKey: string,
    parity?: { text: string; vec: number[] },
    warn: (msg: string) => void = () => {},
  ): Promise<EmbedFn> {
    const cached = this.cache.get(modelKey);
    if (cached) return cached;
    const built = this.build(modelKey, parity, warn);
    this.cache.set(modelKey, built);
    built.catch(() => this.cache.delete(modelKey));
    return built;
  }

  private async build(
    modelKey: string,
    parity: { text: string; vec: number[] } | undefined,
    warn: (msg: string) => void,
  ): Promise<EmbedFn> {
    const basename = modelKey.split('/').pop() ?? modelKey;
    const modelIds = [...new Set([modelKey, `Xenova/${basename}`])];
    const dtypes: Array<'fp32' | 'q8'> = ['fp32', 'q8'];
    let lastError: unknown = new Error('no variants attempted');

    for (const modelId of modelIds) {
      for (const dtype of dtypes) {
        let extractor: RawExtractor;
        try {
          extractor = await this.factory(modelId, { dtype });
        } catch (e) {
          lastError = e;
          continue;
        }
        const embed: EmbedFn = async (text) => {
          let input = text;
          if (input.length > EMBED_MAX_CHARS) {
            console.error(
              `[embedder] input truncated from ${input.length} to ${EMBED_MAX_CHARS} chars for ${modelId} ` +
                `(small embedding models cap out near 512 tokens)`,
            );
            input = input.slice(0, EMBED_MAX_CHARS);
          }
          const out = await extractor(input, { pooling: 'mean', normalize: true });
          return Array.from(out.data as Float32Array);
        };
        if (parity) {
          let computed: number[];
          try {
            computed = await embed(parity.text);
          } catch (e) {
            lastError = e;
            continue;
          }
          if (computed.length !== parity.vec.length) {
            lastError = new Error(
              `${modelId} (${dtype}) produced ${computed.length}-dim vectors, vault has ${parity.vec.length}-dim`,
            );
            continue;
          }
          const cos = cosineSimilarity(computed, parity.vec);
          if (cos < PARITY_WARN_THRESHOLD) {
            warn(
              `embedding parity for ${modelId} (${dtype}) is ${cos.toFixed(4)} (< ${PARITY_WARN_THRESHOLD}); ` +
                `rankings may differ slightly from Smart Connections`,
            );
          }
        }
        return embed;
      }
    }
    throw new EmbedUnavailableError(
      `Could not load embedding model "${modelKey}": ${lastError instanceof Error ? lastError.message : String(lastError)}`,
    );
  }
}
