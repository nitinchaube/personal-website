import { getNoteTree } from '../../lib/notes';
import CategoryGrid from '../../components/notes/CategoryGrid';

export const metadata = {
  title: "Nitin's Desk",
  description: "Things I'm learning, written down.",
};

export default function NotesIndexPage() {
  const tree = getNoteTree();

  return (
    <main className='mx-auto max-w-5xl px-6 py-14 sm:px-8 sm:py-20'>
      <header className='mb-14'>
        <h1 className='notes-card-title text-[44px] sm:text-[52px] leading-[1.05]'>Nitin&apos;s Desk</h1>
        <p className='mt-4 max-w-2xl text-[17px] leading-relaxed text-[color:var(--notes-text-muted)]'>
          A scratchpad of what I&apos;m learning, organized by folder. Each box below is a topic, and the links inside
          are individual notes.
        </p>
      </header>
      <CategoryGrid nodes={tree} />
    </main>
  );
}
