'use client';

import { useEffect, useMemo, useState } from 'react';

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

type Props = {
  selector?: string;
};

const levelFromTag = (tagName: string): number => {
  const n = Number(tagName.replace(/^H/i, ''));
  return Number.isFinite(n) ? n : 2;
};

/** Group h3/h4 headings under the preceding h2 as collapsible sections. */
function buildTocTree(headings: Heading[]): TocEntry[] {
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

function sectionIsActive(section: TocSection, activeId: string): boolean {
  if (section.heading.id === activeId) return true;
  return section.children.some((c) => c.id === activeId);
}

const Toc = ({ selector = '.notes-article' }: Props) => {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string>('');
  const [openSections, setOpenSections] = useState<Set<string>>(new Set());

  const tree = useMemo(() => buildTocTree(headings), [headings]);

  useEffect(() => {
    const root = document.querySelector(selector);
    if (!root) return;
    const nodes = root.querySelectorAll<HTMLHeadingElement>('h1, h2, h3, h4');
    const collected: Heading[] = [];
    nodes.forEach((node) => {
      if (!node.id) return;
      collected.push({
        id: node.id,
        text: node.innerText.replace(/#$/, '').trim(),
        level: levelFromTag(node.tagName),
      });
    });
    setHeadings(collected);

    if (collected.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActiveId(visible[0].target.id);
      },
      { rootMargin: '-72px 0px -70% 0px', threshold: 0 }
    );
    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, [selector]);

  // Keep the h2 section open while its heading or any child h3/h4 is active.
  useEffect(() => {
    if (!activeId) return;
    setOpenSections((prev) => {
      const next = new Set(prev);
      for (const entry of tree) {
        if (entry.kind === 'section' && sectionIsActive(entry, activeId)) {
          next.add(entry.heading.id);
        }
      }
      return next;
    });
  }, [activeId, tree]);

  const linkClass = (id: string) =>
    activeId === id ? 'notes-toc-link is-active' : 'notes-toc-link';

  if (headings.length < 2) return null;

  return (
    <nav className='notes-toc' aria-label='On this page'>
      <p className='notes-toc-heading'>On this page</p>
      <div className='notes-toc-scroll'>
        <ul className='notes-toc-list'>
          {tree.map((entry) => {
            if (entry.kind === 'link') {
              const h = entry.heading;
              return (
                <li key={h.id} className={`notes-toc-item notes-toc-item--h${h.level}`}>
                  <a href={`#${h.id}`} className={linkClass(h.id)}>
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
                      className={linkClass(heading.id)}
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenSections((prev) => new Set(prev).add(heading.id));
                      }}
                    >
                      {heading.text}
                    </a>
                  </summary>
                  <ul className='notes-toc-sublist'>
                    {children.map((child) => (
                      <li key={child.id} className={`notes-toc-item notes-toc-item--h${child.level}`}>
                        <a href={`#${child.id}`} className={linkClass(child.id)}>
                          {child.text}
                        </a>
                      </li>
                    ))}
                  </ul>
                </details>
              </li>
            );
          })}
        </ul>
      </div>
    </nav>
  );
};

export default Toc;
