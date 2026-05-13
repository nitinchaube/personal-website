// Server-only module: reads files from disk. Do not import this from
// client components — use ./notes-shared for types and formatting helpers.
import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';
import readingTime from 'reading-time';

import {
  type CategoryGroup,
  type NoteFrontmatter,
  type NoteFull,
  type NoteSummary,
  type NoteTag,
  type SearchEntry,
  NOTE_TAGS,
  prettyName,
  stripOrderPrefix,
  stripOrderPrefixPath,
} from './notes-shared';

export { NOTE_TAGS, formatDate, prettyName, stripOrderPrefix, stripOrderPrefixPath } from './notes-shared';
export type { CategoryGroup, NoteFrontmatter, NoteFull, NoteSummary, NoteTag, SearchEntry } from './notes-shared';

const NOTES_DIR = path.join(process.cwd(), 'content', 'notes');

/**
 * Walk content/notes/ recursively. Skip `assets/` folders and hidden entries.
 * Entries are sorted alphabetically by raw name so that numeric-prefixed
 * folders/files (e.g. "01-fundamentals", "02-playbooks") drive the discovery
 * order, which downstream becomes the display order of categories and posts.
 */
function walkNoteFiles(dir: string, relDir: string = ''): { absPath: string; relPath: string }[] {
  if (!fs.existsSync(dir)) return [];
  const results: { absPath: string; relPath: string }[] = [];
  const entries = fs
    .readdirSync(dir, { withFileTypes: true })
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    if (entry.name === 'assets') continue;
    const absPath = path.join(dir, entry.name);
    const relPath = relDir ? `${relDir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      results.push(...walkNoteFiles(absPath, relPath));
    } else if (entry.isFile() && /\.(md|mdx)$/i.test(entry.name)) {
      results.push({ absPath, relPath });
    }
  }
  return results;
}

/** Convert a filesystem relPath into a frontend slug, stripping any "01-" / "1_" prefixes. */
function slugFromRelPath(relPath: string): { slug: string; category: string | null; postDir: string } {
  const noExt = relPath.replace(/\.(md|mdx)$/i, '');
  const parts = noExt.split('/').map(stripOrderPrefix);
  const category = parts.length > 1 ? parts[0] : null;
  const postDir = parts.length > 1 ? parts.slice(0, -1).join('/') : '';
  return { slug: parts.join('/'), category, postDir };
}

function readNote(relPath: string): NoteFull | null {
  const filepath = path.join(NOTES_DIR, relPath);
  const raw = fs.readFileSync(filepath, 'utf8');
  const { data, content } = matter(raw);
  if (!data || !data.title || !data.date) {
    console.warn(`[notes] skipping ${relPath}: missing required frontmatter (title, date)`);
    return null;
  }
  if (data.draft) return null;

  const frontmatter = data as NoteFrontmatter;
  const stats = readingTime(content);
  const { slug, category, postDir } = slugFromRelPath(relPath);

  return {
    slug,
    category,
    postDir,
    title: frontmatter.title,
    date: typeof frontmatter.date === 'string' ? frontmatter.date : new Date(frontmatter.date).toISOString().slice(0, 10),
    summary: frontmatter.summary,
    tags: frontmatter.tags ?? [],
    draft: frontmatter.draft ?? false,
    readingTimeText: stats.text,
    content,
  };
}

export function getAllNotes(): NoteSummary[] {
  // walkNoteFiles already returns files in stable alphabetical-by-raw-name
  // order, which honours any "01-foo" / "02-bar" prefix the user added on disk.
  const files = walkNoteFiles(NOTES_DIR);
  const notes = files
    .map(({ relPath }) => readNote(relPath))
    .filter((n): n is NoteFull => n !== null);
  return notes.map(({ content: _content, ...rest }) => rest);
}

export function getNoteBySlug(slug: string): NoteFull | null {
  // The user-visible slug is stripped of numeric prefixes, but the file on
  // disk may carry one (e.g. "01-fundamentals/01-python.mdx"). Walk the tree
  // and match against the stripped form.
  for (const { relPath } of walkNoteFiles(NOTES_DIR)) {
    const noExt = relPath.replace(/\.(md|mdx)$/i, '');
    const stripped = stripOrderPrefixPath(noExt);
    if (stripped === slug) return readNote(relPath);
  }
  return null;
}

export function getAllSlugSegments(): string[][] {
  return getAllNotes().map((n) => n.slug.split('/'));
}

export function getAllTags(): { tag: NoteTag; count: number }[] {
  const counts = new Map<NoteTag, number>();
  for (const note of getAllNotes()) {
    for (const tag of note.tags ?? []) {
      counts.set(tag, (counts.get(tag) ?? 0) + 1);
    }
  }
  return NOTE_TAGS.filter((t) => counts.has(t)).map((tag) => ({ tag, count: counts.get(tag)! }));
}

export function getAdjacentNotes(slug: string): { prev: NoteSummary | null; next: NoteSummary | null } {
  const all = getAllNotes();
  const idx = all.findIndex((n) => n.slug === slug);
  if (idx === -1) return { prev: null, next: null };
  return {
    prev: idx < all.length - 1 ? all[idx + 1] : null,
    next: idx > 0 ? all[idx - 1] : null,
  };
}

/** Returns the data shipped to the client-side search bar. Build-time only. */
export function getSearchEntries(): SearchEntry[] {
  return getAllNotes().map((n) => ({
    slug: n.slug,
    title: n.title,
    summary: n.summary,
    tags: n.tags,
    category: n.category,
    categoryDisplay: n.category ? prettyName(n.category) : null,
  }));
}

/**
 * Returns posts grouped by their top-level folder (category).
 * Categories appear in alphabetical order; root-level (uncategorized) posts go last in a "General" bucket.
 * Inside each group, posts are sorted newest-first.
 */
export function getCategoryGroups(): CategoryGroup[] {
  const all = getAllNotes();
  const map = new Map<string | null, NoteSummary[]>();
  // Inserting in walk order means the Map's key iteration preserves the
  // numeric-prefix-driven category ordering for free.
  for (const note of all) {
    const key = note.category ?? null;
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(note);
  }
  const categories = Array.from(map.keys()).filter((c): c is string => c !== null);
  const groups: CategoryGroup[] = categories.map((cat) => ({
    category: cat,
    displayName: prettyName(cat),
    posts: map.get(cat)!,
  }));
  // Root-level (uncategorised) posts always go last in a "General" bucket.
  if (map.has(null)) {
    groups.push({ category: null, displayName: 'General', posts: map.get(null)! });
  }
  return groups;
}
