import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { EmbeddingGemmaIndex } from '../src/embedding-gemma-index.js';

describe('EmbeddingGemmaIndex', () => {
  it('finds stored-vector neighbors and filters stale paths', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'scmcp-gemma-'));
    try {
      fs.mkdirSync(path.join(tmp, '.smart-env'));
      const entry = (notePath: string, vec: number[]) => ({
        vec,
        note_path: notePath,
        block_key: '__full__',
        block_type: 'full',
        char_length: 100,
        hash: notePath,
        updated_at: 1,
      });
      fs.writeFileSync(
        path.join(tmp, '.smart-env', 'embedding-index.json'),
        JSON.stringify({
          model: 'onnx-community/embeddinggemma-300m-ONNX',
          dimension: 768,
          version: 4,
          created_at: 1,
          updated_at: 1,
          entries: {
            'A.md': entry('A.md', [1, 0, 0]),
            'B.md': entry('B.md', [0.9, 0.1, 0]),
            'Deleted.md': entry('Deleted.md', [1, 0, 0]),
          },
        }),
      );
      const index = new EmbeddingGemmaIndex(tmp);
      const results = index.similarByPath('A.md', new Set(['A.md', 'B.md']), 5, 0.5);
      expect(results).toEqual([{ path: 'B.md', score: expect.any(Number) }]);
      expect(index.similarByPath('Missing.md', new Set(['A.md', 'B.md']), 5, 0.5)).toBeNull();
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it('accepts the v1 legacy hash without loading a model or re-embedding', async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'scmcp-gemma-v1-'));
    try {
      fs.mkdirSync(path.join(tmp, '.smart-env'));
      const content = 'unchanged legacy content long enough for incremental indexing';
      let value = 0;
      for (let i = 0; i < content.length; i++) {
        value = ((value << 5) - value + content.charCodeAt(i)) | 0;
      }
      fs.writeFileSync(
        path.join(tmp, '.smart-env', 'embedding-index.json'),
        JSON.stringify({
          model: 'onnx-community/embeddinggemma-300m-ONNX',
          dimension: 768,
          version: 4,
          created_at: 1,
          updated_at: 1,
          entries: {
            'Legacy.md': {
              vec: [1, 0, 0],
              note_path: 'Legacy.md',
              block_key: '__full__',
              block_type: 'full',
              char_length: content.length,
              hash: `${content.length}:${value}`,
              updated_at: 1,
            },
          },
        }),
      );
      const index = new EmbeddingGemmaIndex(tmp);
      await expect(index.sync(['Legacy.md'], () => content)).resolves.toMatchObject({
        updated: 0,
        unchanged: 1,
        deleted: 0,
      });
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });
});
