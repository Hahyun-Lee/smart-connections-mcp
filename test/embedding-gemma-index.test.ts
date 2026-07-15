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
});
