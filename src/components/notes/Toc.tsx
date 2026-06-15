'use client';

import { useEffect, useState } from 'react';

type Heading = {
  id: string;
  text: string;
  level: number;
};

type Props = {
  selector?: string;
};

const levelFromTag = (tagName: string): number => {
  const n = Number(tagName.replace(/^H/i, ''));
  return Number.isFinite(n) ? n : 2;
};

const Toc = ({ selector = '.notes-article' }: Props) => {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string>('');

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

  if (headings.length < 2) return null;

  return (
    <nav className='notes-toc' aria-label='On this page'>
      <p className='notes-toc-heading'>On this page</p>
      <div className='notes-toc-scroll'>
        <ul className='notes-toc-list'>
          {headings.map((h) => (
            <li key={h.id} className={`notes-toc-item notes-toc-item--h${h.level}`}>
              <a
                href={`#${h.id}`}
                className={activeId === h.id ? 'notes-toc-link is-active' : 'notes-toc-link'}
              >
                {h.text}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
};

export default Toc;
