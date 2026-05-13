'use client';

import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { SearchEntry } from '../../lib/notes-shared';
import { withBase } from '../../lib/site';

type Props = {
  entries: SearchEntry[];
};

const MAX_RESULTS = 8;

function scoreEntry(entry: SearchEntry, query: string): number {
  const q = query.toLowerCase();
  const title = entry.title.toLowerCase();
  if (title === q) return 100;
  if (title.startsWith(q)) return 80;
  if (title.includes(q)) return 60;
  if (entry.summary?.toLowerCase().includes(q)) return 40;
  if (entry.tags?.some((t) => t.toLowerCase().includes(q))) return 30;
  if (entry.category?.toLowerCase().includes(q)) return 20;
  if (entry.categoryDisplay?.toLowerCase().includes(q)) return 20;
  return 0;
}

const SearchBar = ({ entries }: Props) => {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const results = useMemo(() => {
    const q = query.trim();
    if (!q) return [];
    return entries
      .map((e) => ({ entry: e, score: scoreEntry(e, q) }))
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_RESULTS)
      .map((r) => r.entry);
  }, [entries, query]);

  // Reset highlighted result whenever the result set changes.
  useEffect(() => {
    setActiveIdx(0);
  }, [results]);

  // Cmd/Ctrl+K focuses the input from anywhere on the page.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      } else if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
        inputRef.current?.blur();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen]);

  // Close the dropdown when clicking outside.
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target as Node)) setIsOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (results.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIdx((i) => (i + 1) % results.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx((i) => (i - 1 + results.length) % results.length);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const target = results[activeIdx];
      if (target) {
        window.location.href = withBase(`/notes/${target.slug}`);
      }
    }
  };

  return (
    <div ref={containerRef} className='notes-search'>
      <label className='notes-search-input-wrap'>
        <svg className='notes-search-icon' viewBox='0 0 24 24' aria-hidden='true'>
          <circle cx='11' cy='11' r='7' fill='none' stroke='currentColor' strokeWidth='1.75' />
          <line x1='16.5' y1='16.5' x2='21' y2='21' stroke='currentColor' strokeWidth='1.75' strokeLinecap='round' />
        </svg>
        <input
          ref={inputRef}
          type='search'
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={onKeyDown}
          placeholder='Search notes…'
          className='notes-search-input'
          aria-label='Search notes'
          aria-autocomplete='list'
          aria-expanded={isOpen && results.length > 0}
        />
        <kbd className='notes-search-kbd' aria-hidden='true'>⌘K</kbd>
      </label>

      {isOpen && query.trim().length > 0 && (
        <div className='notes-search-dropdown' role='listbox'>
          {results.length === 0 ? (
            <div className='notes-search-empty'>No matches for &ldquo;{query}&rdquo;</div>
          ) : (
            <ul>
              {results.map((r, i) => (
                <li key={r.slug} role='option' aria-selected={i === activeIdx}>
                  <Link
                    href={`/notes/${r.slug}`}
                    className='notes-search-result'
                    data-active={i === activeIdx}
                    onMouseEnter={() => setActiveIdx(i)}
                    onClick={() => setIsOpen(false)}
                  >
                    <span className='notes-search-result-title'>{r.title}</span>
                    {(r.categoryDisplay || r.summary) && (
                      <span className='notes-search-result-meta'>
                        {r.categoryDisplay && <span className='notes-search-result-cat'>{r.categoryDisplay}</span>}
                        {r.categoryDisplay && r.summary && <span className='notes-search-result-sep'> · </span>}
                        {r.summary && <span className='notes-search-result-summary'>{r.summary}</span>}
                      </span>
                    )}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchBar;
