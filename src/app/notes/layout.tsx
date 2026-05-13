import '../globals.css';
import '../../styles/notes.css';
import 'katex/dist/katex.min.css';
import NotesNav from '../../components/notes/NotesNav';

export const metadata = {
  title: "Nitin's Desk",
  description: "Things I'm learning, written down.",
};

export default function NotesLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className='notes-root'>
      <NotesNav />
      {children}
    </div>
  );
}
