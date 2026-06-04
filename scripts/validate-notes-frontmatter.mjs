#!/usr/bin/env node
// Fail the build early when a note's YAML frontmatter is invalid.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const NOTES_DIR = path.join(__dirname, '..', 'content', 'notes');

function walk(dir, relDir = '') {
  const entries = fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
  const files = [];
  for (const entry of entries) {
    if (entry.name.startsWith('.') || entry.name === 'assets') continue;
    const abs = path.join(dir, entry.name);
    const rel = relDir ? `${relDir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) files.push(...walk(abs, rel));
    else if (/\.(md|mdx)$/i.test(entry.name)) files.push(rel);
  }
  return files;
}

let failed = false;
for (const rel of walk(NOTES_DIR)) {
  const filepath = path.join(NOTES_DIR, rel);
  const raw = fs.readFileSync(filepath, 'utf8');
  try {
    const { data } = matter(raw);
    if (!data?.title || !data?.date) {
      console.error(`[validate-notes-frontmatter] ${rel}: missing required frontmatter (title, date)`);
      failed = true;
      continue;
    }
    if (typeof data.title === 'string' && /[\r\n]/.test(data.title)) {
      console.error(`[validate-notes-frontmatter] ${rel}: title must be a single line`);
      failed = true;
    }
  } catch (err) {
    console.error(`[validate-notes-frontmatter] ${rel}: ${err.message}`);
    failed = true;
  }
}

if (failed) process.exit(1);
