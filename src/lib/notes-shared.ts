// Client-safe types and helpers. No `node:fs`/`node:path` imports here so that
// React client components can use these without webpack errors.

export const NOTE_TAGS = ['Foundations', 'Internals', 'Snippets', 'Readings', 'Playbooks', 'Lab'] as const;
export type NoteTag = (typeof NOTE_TAGS)[number];

export type NoteFrontmatter = {
  title: string;
  date: string; // ISO yyyy-mm-dd
  summary?: string;
  tags?: NoteTag[];
  draft?: boolean;
};

export type NoteSummary = NoteFrontmatter & {
  /** Slug includes folder path. e.g. "fundamentals/python" or "welcome". */
  slug: string;
  /** Top-level folder name, or null for root-level posts. */
  category: string | null;
  /** Directory part of the slug (everything except the last segment), or "" for root posts. */
  postDir: string;
  readingTimeText: string;
};

export type NoteContentFormat = 'md' | 'mdx';

export type NoteFull = NoteSummary & {
  content: string;
  /** `md` disables JSX parsing so LaTeX in `.md` notes is safe; `mdx` keeps MDX features. */
  contentFormat: NoteContentFormat;
};

export type CategoryGroup = {
  /** Raw folder slug, e.g. "fundamentals". Null for the catch-all bucket of root-level posts. */
  category: string | null;
  /** Pretty display label, e.g. "Fundamentals". */
  displayName: string;
  posts: NoteSummary[];
};

/** Lightweight payload used by the in-header search. Safe to ship to the client. */
export type SearchEntry = {
  slug: string;
  title: string;
  summary?: string;
  tags?: NoteTag[];
  category: string | null;
  categoryDisplay: string | null;
};

export function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}

/** "deep-learning" -> "Deep Learning"; "rlhf_notes" -> "Rlhf Notes"; "ML" stays "ML". */
export function prettyName(raw: string): string {
  return raw
    .split(/[-_]+/)
    .map((part) => (part.length === 0 ? part : part[0].toUpperCase() + part.slice(1)))
    .join(' ');
}

/**
 * Strip a leading numeric ordering prefix from a single path segment.
 *   "01-fundamentals"  -> "fundamentals"
 *   "1_python"         -> "python"
 *   "100-deep-learning"-> "deep-learning"
 *   "welcome"          -> "welcome"      (no-op)
 *
 * Used so that folder/file names can carry an ordering hint on disk without
 * leaking into URLs or display names.
 */
export function stripOrderPrefix(segment: string): string {
  return segment.replace(/^\d+[-_]/, '');
}

/** Apply stripOrderPrefix to every segment of a slug-like path. */
export function stripOrderPrefixPath(p: string): string {
  return p.split('/').map(stripOrderPrefix).join('/');
}
