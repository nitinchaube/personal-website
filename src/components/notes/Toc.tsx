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

const Toc = ({ selector = '.notes-article' }: Props) => {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string>('');

  useEffect(() => {
    const root = document.querySelector(selector);
    if (!root) return;
    const nodes = root.querySelectorAll<HTMLHeadingElement>('h2, h3');
    const collected: Heading[] = [];
    nodes.forEach((node) => {
      if (!node.id) return;
      collected.push({
        id: node.id,
        text: node.innerText.replace(/#$/, '').trim(),
        level: node.tagName === 'H2' ? 2 : 3,
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
    <nav className='notes-toc'>
      <p className='notes-toc-heading'>On this page</p>
      <div className='notes-toc-scroll'>
        <ul className='list-none'>
          {headings.map((h) => (
            <li key={h.id} style={{ paddingLeft: h.level === 3 ? '0.85rem' : 0 }}>
              <a
                href={`#${h.id}`}
                style={{
                  color: activeId === h.id ? 'var(--notes-text)' : undefined,
                  fontWeight: activeId === h.id ? 600 : 400,
                }}
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
