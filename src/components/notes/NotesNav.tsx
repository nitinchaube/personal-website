import Link from 'next/link';
import { getSearchEntries } from '../../lib/notes';
import SearchBar from './SearchBar';

const NotesNav = () => {
  const entries = getSearchEntries();

  return (
    <header className='notes-nav'>
      <div className='mx-auto flex w-full max-w-[1400px] items-center gap-4 px-6 py-6 sm:gap-8 sm:px-10 sm:py-7'>
        <Link href='/' className='notes-nav-link notes-wordmark shrink-0 text-[18px] font-semibold sm:text-[40px]'>
          Portfolio
        </Link>
        <div className='ml-auto flex flex-1 items-center justify-end gap-4 sm:gap-6'>
          <div className='w-full max-w-md'>
            <SearchBar entries={entries} />
          </div>
          <nav className='shrink-0'>
            <Link href='/notes' className='notes-nav-link text-[16px]'>
              Home
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default NotesNav;
