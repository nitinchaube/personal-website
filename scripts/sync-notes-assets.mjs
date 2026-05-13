#!/usr/bin/env node
// Mirrors any "assets/" directory under content/notes/ into public/notes-assets/,
// preserving its path relative to content/notes/.
//
// Examples:
//   content/notes/assets/welcome/foo.png
//     -> public/notes-assets/welcome/foo.png
//   content/notes/fundamentals/assets/python/diagram.png
//     -> public/notes-assets/fundamentals/python/diagram.png
//
// This lets Cursor's "paste image into markdown" feature save files next to
// the post they belong to, while still serving them from a single public path.

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const SRC = path.join(ROOT, 'content', 'notes');
const DEST = path.join(ROOT, 'public', 'notes-assets');

/**
 * Strip a leading numeric ordering prefix from a single path segment so that
 * a folder named "01-fundamentals" on disk maps to URL segment "fundamentals".
 * Kept in sync with stripOrderPrefix in src/lib/notes-shared.ts.
 */
function stripOrderPrefix(name) {
  return name.replace(/^\d+[-_]/, '');
}

async function exists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function rmrf(target) {
  if (await exists(target)) {
    await fs.rm(target, { recursive: true, force: true });
  }
}

/** Copy src → dest, stripping numeric prefixes from every directory name. */
async function copyDir(src, dest) {
  await fs.mkdir(dest, { recursive: true });
  const entries = await fs.readdir(src, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const s = path.join(src, entry.name);
    const destName = entry.isDirectory() ? stripOrderPrefix(entry.name) : entry.name;
    const d = path.join(dest, destName);
    if (entry.isDirectory()) {
      await copyDir(s, d);
    } else if (entry.isFile()) {
      await fs.copyFile(s, d);
    }
  }
}

let copied = 0;

async function walk(dir, relParent) {
  if (!(await exists(dir))) return;
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    if (!entry.isDirectory()) continue;
    const abs = path.join(dir, entry.name);
    if (entry.name === 'assets') {
      // Mirror contents to DEST + relParent (already prefix-stripped)
      const target = relParent ? path.join(DEST, relParent) : DEST;
      await copyDir(abs, target);
      copied += 1;
    } else {
      // Strip any "01-" prefix from this folder name when constructing the
      // public-facing destination path.
      const stripped = stripOrderPrefix(entry.name);
      await walk(abs, relParent ? path.join(relParent, stripped) : stripped);
    }
  }
}

async function main() {
  if (!(await exists(SRC))) {
    console.log('[sync-notes-assets] no content/notes directory; skipping');
    return;
  }
  await rmrf(DEST);
  await walk(SRC, '');
  if (copied === 0) {
    console.log('[sync-notes-assets] no assets/ folders found; nothing to sync');
  } else {
    console.log(`[sync-notes-assets] mirrored ${copied} assets folder(s) into ${path.relative(ROOT, DEST)}`);
  }
}

main().catch((err) => {
  console.error('[sync-notes-assets] failed:', err);
  process.exit(1);
});
