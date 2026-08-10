'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';

type Heading = {
  id: string;
  text: string;
  level: number;
};

type TocLink = {
  kind: 'link';
  heading: Heading;
};

type TocSection = {
  kind: 'section';
  heading: Heading;
  children: Heading[];
};

type TocEntry = TocLink | TocSection;

type TocGroup = {
  kind: 'group';
  heading: Heading;
  children: TocEntry[];
};

type TocNode = TocGroup | TocEntry;

type Props = {
  selector?: string;
  /** `sheet` drops the sidebar chrome so the list can live inside the mobile bottom sheet. */
  variant?: 'sidebar' | 'sheet';
  /** Fired after a heading jump, so the mobile sheet can dismiss itself. */
  onNavigate?: () => void;
};

const levelFromTag = (tagName: string): number => {
  const n = Number(tagName.replace(/^H/i, ''));
  return Number.isFinite(n) ? n : 2;
};

/** Group h3/h4 headings under the preceding h2 as collapsible sections. */
function buildEntries(headings: Heading[]): TocEntry[] {
  const entries: TocEntry[] = [];
  let i = 0;

  while (i < headings.length) {
    const h = headings[i];

    if (h.level === 2) {
      const children: Heading[] = [];
      let j = i + 1;
      while (j < headings.length && headings[j].level > 2) {
        children.push(headings[j]);
        j++;
      }
      if (children.length > 0) {
        entries.push({ kind: 'section', heading: h, children });
        i = j;
        continue;
      }
    }

    entries.push({ kind: 'link', heading: h });
    i++;
  }

  return entries;
}

/**
 * Build a two-level tree: h1 headings become group labels, and everything
 * beneath them (h2 papers, plus their nested h3/h4 sections) is nested inside.
 * Headings that appear before any h1 are kept as top-level entries.
 */
function buildTocTree(headings: Heading[]): TocNode[] {
  const nodes: TocNode[] = [];
  let i = 0;

  while (i < headings.length) {
    const h = headings[i];

    if (h.level === 1) {
      const groupHeadings: Heading[] = [];
      let j = i + 1;
      while (j < headings.length && headings[j].level > 1) {
        groupHeadings.push(headings[j]);
        j++;
      }
      nodes.push({ kind: 'group', heading: h, children: buildEntries(groupHeadings) });
      i = j;
      continue;
    }

    // No parent h1: collect the run up to the next h1 and keep it top-level.
    const run: Heading[] = [];
    let j = i;
    while (j < headings.length && headings[j].level > 1) {
      run.push(headings[j]);
      j++;
    }
    for (const entry of buildEntries(run)) nodes.push(entry);
    i = j;
  }

  return nodes;
}

function sectionIsActive(section: TocSection, activeId: string): boolean {
  if (section.heading.id === activeId) return true;
  return section.children.some((c) => c.id === activeId);
}

/**
 * Given the active heading id, return every group/section id that should be
 * expanded so the active heading is visible: the group that contains it, plus
 * the specific h2 section (if the active heading is an h3/h4 inside one).
 */
function idsToOpenFor(nodes: TocNode[], activeId: string): string[] {
  const ids: string[] = [];

  const scanEntries = (entries: TocEntry[]): boolean => {
    let anyActive = false;
    for (const entry of entries) {
      if (entry.kind === 'link') {
        if (entry.heading.id === activeId) anyActive = true;
      } else if (sectionIsActive(entry, activeId)) {
        ids.push(entry.heading.id);
        anyActive = true;
      }
    }
    return anyActive;
  };

  for (const node of nodes) {
    if (node.kind === 'group') {
      const childActive = scanEntries(node.children);
      if (childActive || node.heading.id === activeId) ids.push(node.heading.id);
    } else if (node.kind === 'section' && sectionIsActive(node, activeId)) {
      ids.push(node.heading.id);
    }
  }

  return ids;
}

const Toc = ({ selector = '.notes-article', variant = 'sidebar', onNavigate }: Props) => {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [openSections, setOpenSections] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Suppress scroll-driven active updates while a click-initiated smooth
  // scroll is in flight, so the highlight doesn't flicker through sections.
  const jumpTargetRef = useRef<string | null>(null);

  const tree = useMemo(() => buildTocTree(headings), [headings]);

  useEffect(() => {
    const root = document.querySelector<HTMLElement>(selector);
    if (!root) return;

    const nodes = Array.from(root.querySelectorAll<HTMLHeadingElement>('h1, h2, h3, h4')).filter(
      (node) => node.id
    );
    setHeadings(
      nodes.map((node) => ({
        id: node.id,
        text: node.innerText.replace(/#$/, '').trim(),
        level: levelFromTag(node.tagName),
      }))
    );

    if (nodes.length === 0) return;

    let ticking = false;

    const update = () => {
      ticking = false;

      // Active heading = the last one whose top has passed the header line.
      const OFFSET = 96;
      let current = nodes[0].id;
      for (const el of nodes) {
        if (el.getBoundingClientRect().top <= OFFSET) current = el.id;
        else break;
      }
      // At the very bottom of the page, highlight the final heading.
      const doc = document.documentElement;
      if (window.innerHeight + window.scrollY >= doc.scrollHeight - 2) {
        current = nodes[nodes.length - 1].id;
      }

      if (jumpTargetRef.current) {
        if (current === jumpTargetRef.current) jumpTargetRef.current = null;
      } else {
        setActiveId(current);
      }

      // Reading progress through the article.
      const rect = root.getBoundingClientRect();
      const total = rect.height - window.innerHeight;
      setProgress(total > 0 ? Math.min(1, Math.max(0, -rect.top / total)) : 1);
    };

    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    };

    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, [selector]);

  // Groups (h1) and sections (h2) start collapsed. Scrolling into a section
  // expands its parent group + the section itself so the active item shows.
  useEffect(() => {
    if (!activeId) return;
    const ids = idsToOpenFor(tree, activeId);
    if (ids.length === 0) return;
    setOpenSections((prev) => {
      if (ids.every((id) => prev.has(id))) return prev;
      const next = new Set(prev);
      ids.forEach((id) => next.add(id));
      return next;
    });
  }, [activeId, tree]);

  // Keep the active item visible inside the TOC's own scroll area.
  useEffect(() => {
    if (!activeId) return;
    const container = scrollRef.current;
    if (!container) return;
    const raf = requestAnimationFrame(() => {
      const link = container.querySelector<HTMLElement>(`[data-toc-id="${activeId}"]`);
      if (!link) return;
      const cr = container.getBoundingClientRect();
      const lr = link.getBoundingClientRect();
      const PAD = 12;
      if (lr.top < cr.top + PAD) {
        container.scrollTop += lr.top - (cr.top + PAD);
      } else if (lr.bottom > cr.bottom - PAD) {
        container.scrollTop += lr.bottom - (cr.bottom - PAD);
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [activeId, openSections]);

  const handleJump = (e: ReactMouseEvent<HTMLAnchorElement>, id: string, expandId?: string) => {
    e.preventDefault();
    e.stopPropagation();
    const el = document.getElementById(id);
    if (!el) return;
    if (expandId) setOpenSections((prev) => new Set(prev).add(expandId));
    jumpTargetRef.current = id;
    setActiveId(id);
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    window.history.replaceState(null, '', `#${id}`);
    onNavigate?.();
  };

  const linkClass = (id: string) =>
    activeId === id ? 'notes-toc-link is-active' : 'notes-toc-link';

  const renderEntry = (entry: TocEntry) => {
    if (entry.kind === 'link') {
      const h = entry.heading;
      return (
        <li key={h.id} className={`notes-toc-item notes-toc-item--h${h.level}`}>
          <a
            href={`#${h.id}`}
            data-toc-id={h.id}
            className={linkClass(h.id)}
            onClick={(e) => handleJump(e, h.id)}
          >
            {h.text}
          </a>
        </li>
      );
    }

    const { heading, children } = entry;
    const isOpen = openSections.has(heading.id);

    return (
      <li key={heading.id} className='notes-toc-item notes-toc-item--section'>
        <details
          className='notes-toc-section'
          open={isOpen}
          onToggle={(e) => {
            const open = e.currentTarget.open;
            setOpenSections((prev) => {
              if (prev.has(heading.id) === open) return prev;
              const next = new Set(prev);
              if (open) next.add(heading.id);
              else next.delete(heading.id);
              return next;
            });
          }}
        >
          <summary className='notes-toc-section-summary'>
            <a
              href={`#${heading.id}`}
              data-toc-id={heading.id}
              className={linkClass(heading.id)}
              onClick={(e) => handleJump(e, heading.id, heading.id)}
            >
              {heading.text}
            </a>
          </summary>
          <ul className='notes-toc-sublist'>
            {children.map((child) => (
              <li key={child.id} className={`notes-toc-item notes-toc-item--h${child.level}`}>
                <a
                  href={`#${child.id}`}
                  data-toc-id={child.id}
                  className={linkClass(child.id)}
                  onClick={(e) => handleJump(e, child.id)}
                >
                  {child.text}
                </a>
              </li>
            ))}
          </ul>
        </details>
      </li>
    );
  };

  if (headings.length < 2) return null;

  return (
    <nav
      className={variant === 'sheet' ? 'notes-toc notes-toc--sheet' : 'notes-toc'}
      aria-label='On this page'
    >
      <p className='notes-toc-heading'>On this page</p>
      <div
        className='notes-toc-progress'
        role='progressbar'
        aria-label='Reading progress'
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress * 100)}
      >
        <div className='notes-toc-progress-fill' style={{ width: `${progress * 100}%` }} />
      </div>
      <div className='notes-toc-scroll' ref={scrollRef}>
        <ul className='notes-toc-list'>
          {tree.map((node) => {
            if (node.kind === 'group') {
              // A childless h1 (e.g. the page title) is just a plain link.
              if (node.children.length === 0) {
                return (
                  <li key={node.heading.id} className='notes-toc-item notes-toc-item--h1'>
                    <a
                      href={`#${node.heading.id}`}
                      data-toc-id={node.heading.id}
                      className={linkClass(node.heading.id)}
                      onClick={(e) => handleJump(e, node.heading.id)}
                    >
                      {node.heading.text}
                    </a>
                  </li>
                );
              }

              const isOpen = openSections.has(node.heading.id);

              return (
                <li key={node.heading.id} className='notes-toc-item notes-toc-item--group'>
                  <details
                    className='notes-toc-group'
                    open={isOpen}
                    onToggle={(e) => {
                      const open = e.currentTarget.open;
                      setOpenSections((prev) => {
                        if (prev.has(node.heading.id) === open) return prev;
                        const next = new Set(prev);
                        if (open) next.add(node.heading.id);
                        else next.delete(node.heading.id);
                        return next;
                      });
                    }}
                  >
                    <summary className='notes-toc-group-summary'>
                      <a
                        href={`#${node.heading.id}`}
                        data-toc-id={node.heading.id}
                        className={
                          activeId === node.heading.id
                            ? 'notes-toc-group-label is-active'
                            : 'notes-toc-group-label'
                        }
                        onClick={(e) => handleJump(e, node.heading.id, node.heading.id)}
                      >
                        {node.heading.text}
                      </a>
                    </summary>
                    <ul className='notes-toc-list notes-toc-group-children'>
                      {node.children.map(renderEntry)}
                    </ul>
                  </details>
                </li>
              );
            }

            return renderEntry(node);
          })}
        </ul>
      </div>
      <button
        type='button'
        className='notes-toc-top'
        onClick={() => {
          window.scrollTo({ top: 0, behavior: 'smooth' });
          onNavigate?.();
        }}
      >
        ↑ Back to top
      </button>
    </nav>
  );
};

export default Toc;
