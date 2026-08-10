'use client';

import { useEffect, useState } from 'react';

import Toc from './Toc';

type Props = {
  selector?: string;
};

/**
 * Below `lg` the right-rail TOC is hidden, so headings are unreachable on a phone.
 * This puts them behind a floating button that opens a bottom sheet, which stays
 * reachable from anywhere in the article (an inline TOC at the top would not).
 */
const MobileToc = ({ selector = '.notes-article' }: Props) => {
  const [open, setOpen] = useState(false);
  const [hasHeadings, setHasHeadings] = useState(false);

  // Match Toc's own bail-out: nothing to jump between on a note with < 2 headings.
  useEffect(() => {
    const root = document.querySelector<HTMLElement>(selector);
    if (!root) return;
    const count = Array.from(root.querySelectorAll<HTMLHeadingElement>('h1, h2, h3, h4')).filter(
      (node) => node.id
    ).length;
    setHasHeadings(count >= 2);
  }, [selector]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open]);

  if (!hasHeadings) return null;

  return (
    <div className='lg:hidden'>
      <button
        type='button'
        className='notes-toc-fab'
        aria-expanded={open}
        aria-label='On this page'
        onClick={() => setOpen((v) => !v)}
      >
        <svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'>
          <path
            d='M2 3.5h12M2 8h12M2 12.5h8'
            fill='none'
            stroke='currentColor'
            strokeWidth='1.6'
            strokeLinecap='round'
          />
        </svg>
        Contents
      </button>

      {open && (
        <>
          <div className='notes-toc-backdrop' onClick={() => setOpen(false)} />
          <div className='notes-toc-sheet' role='dialog' aria-modal='true' aria-label='On this page'>
            <button
              type='button'
              className='notes-toc-sheet-close'
              aria-label='Close'
              onClick={() => setOpen(false)}
            >
              ✕
            </button>
            <Toc selector={selector} variant='sheet' onNavigate={() => setOpen(false)} />
          </div>
        </>
      )}
    </div>
  );
};

export default MobileToc;
