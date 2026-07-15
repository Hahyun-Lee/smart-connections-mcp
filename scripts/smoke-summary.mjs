#!/usr/bin/env node

/** Privacy-safe stdio smoke: prints status and timing, never note paths/content. */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

if (!process.env.SMART_VAULT_PATH && !process.env.SMART_VAULT_PATHS) {
  console.error('SMART_VAULT_PATH is required');
  process.exit(1);
}

const here = path.dirname(fileURLToPath(import.meta.url));
const query = process.argv.slice(2).join(' ') || 'memory lifecycle and agent governance';
const transport = new StdioClientTransport({
  command: process.execPath,
  args: [path.resolve(here, '../dist/index.js')],
  env: { ...process.env },
  stderr: 'ignore',
});
const client = new Client({ name: 'privacy-safe-smoke', version: '1.0.0' });
try {
  await client.connect(transport);
  const { tools } = await client.listTools();
  const statsResponse = await client.callTool({ name: 'get_stats', arguments: {} });
  const statsText = statsResponse.content?.find((item) => item.type === 'text')?.text ?? '{}';
  const statsBody = JSON.parse(statsText);
  const vaultStats = Array.isArray(statsBody.vaults) ? statsBody.vaults : [];
  const callSearch = () => client.callTool({
      name: 'search_notes',
      arguments: { query, limit: 10, threshold: 0 },
    });
  const coldStarted = performance.now();
  const response = await callSearch();
  const coldSearchMs = performance.now() - coldStarted;
  const warmStarted = performance.now();
  await callSearch();
  const warmSearchMs = performance.now() - warmStarted;
  const text = response.content?.find((item) => item.type === 'text')?.text ?? '{}';
  const body = JSON.parse(text);
  const results = Array.isArray(body.results) ? body.results : [];
  const retrieval = [...new Set(results.flatMap((item) => item.retrieval ?? []))].sort();
  console.log(JSON.stringify({
    tools: tools.length,
    mode: body.mode,
    profile: body.profile,
    results: results.length,
    retrieval,
    warning: Boolean(body.warning),
    diskNotes: vaultStats.reduce((sum, item) => sum + Number(item.diskNotes ?? 0), 0),
    pluginIndexedNotes: vaultStats.reduce((sum, item) => sum + Number(item.pluginIndexedNotes ?? 0), 0),
    diskOnlyNotes: vaultStats.reduce((sum, item) => sum + Number(item.diskOnlyNotes ?? 0), 0),
    staleGemmaNotes: vaultStats.reduce(
      (sum, item) => sum + Number(item.embeddingGemma?.staleNotes ?? 0),
      0,
    ),
    coldSearchMs: Math.round(coldSearchMs * 10) / 10,
    warmSearchMs: Math.round(warmSearchMs * 10) / 10,
  }));
} finally {
  await client.close();
}
