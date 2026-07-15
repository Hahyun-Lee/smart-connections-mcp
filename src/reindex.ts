#!/usr/bin/env node

/** Build or incrementally refresh the optional EmbeddingGemma index. */

import { parseVaultPaths, VaultRegistry } from './vault-registry.js';

const paths = parseVaultPaths();
if (paths.length === 0) {
  console.error('SMART_VAULT_PATH is required (comma-separated paths are supported).');
  process.exit(1);
}

const requestedVault = process.argv[2];
const registry = VaultRegistry.fromPaths(paths);
const vaults = registry.byName(requestedVault);
for (const vault of vaults) {
  console.error(`[${vault.name}] syncing EmbeddingGemma over ${vault.notePaths().length} disk notes`);
  const result = await vault.gemma.sync(vault.notePaths(), (notePath) => vault.readNote(notePath));
  console.log(JSON.stringify({ vault: vault.name, ...result, stats: vault.gemma.stats(vault.notePathSet()) }));
}
