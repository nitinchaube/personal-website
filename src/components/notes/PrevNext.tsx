import Link from 'next/link';
import type { NoteSummary } from '../../lib/notes-shared';

type Props = {
  prev: NoteSummary | null;
  next: NoteSummary | null;
};

const PrevNext = ({ prev, next }: Props) => {
  if (!prev && !next) return null;
  return (
    <nav className='mt-16 grid grid-cols-1 gap-6 sm:grid-cols-2'>
      {prev ? (
        <Link href={`/notes/${prev.slug}`} className='block rounded border border-[color:var(--notes-border)] p-4 no-underline transition-colors hover:border-[color:var(--notes-link)]'>
          <p className='notes-meta mb-1'>← Older</p>
          <p className='notes-card-title text-[17px] leading-snug'>{prev.title}</p>
        </Link>
      ) : (
        <span />
      )}
      {next ? (
        <Link
          href={`/notes/${next.slug}`}
          className='block rounded border border-[color:var(--notes-border)] p-4 text-right no-underline transition-colors hover:border-[color:var(--notes-link)]'
        >
          <p className='notes-meta mb-1'>Newer →</p>
          <p className='notes-card-title text-[17px] leading-snug'>{next.title}</p>
        </Link>
      ) : (
        <span />
      )}
    </nav>
  );
};

export default PrevNext;
