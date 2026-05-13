import { formatDate } from '../../lib/notes-shared';
import type { NoteSummary } from '../../lib/notes-shared';

type Props = {
  note: NoteSummary;
};

const PostHeader = ({ note }: Props) => {
  return (
    <header className='mb-10 sm:mb-12'>
      <h1 className='notes-card-title text-[34px] sm:text-[40px] leading-tight'>{note.title}</h1>
      <p className='notes-meta mt-3'>
        {formatDate(note.date)}
        <span className='mx-2'>·</span>
        {note.readingTimeText}
        {note.tags && note.tags.length > 0 && (
          <>
            <span className='mx-2'>·</span>
            {note.tags.join(' · ')}
          </>
        )}
      </p>
      {note.summary && (
        <p className='mt-4 text-[17.5px] italic leading-relaxed text-[color:var(--notes-text-muted)]'>
          {note.summary}
        </p>
      )}
    </header>
  );
};

export default PostHeader;
