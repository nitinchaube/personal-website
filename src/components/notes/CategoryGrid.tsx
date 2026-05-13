import Link from 'next/link';
import type { CategoryGroup } from '../../lib/notes-shared';

type Props = {
  groups: CategoryGroup[];
};

const CategoryGrid = ({ groups }: Props) => {
  if (groups.length === 0) {
    return <p className='notes-meta'>No notes yet. Drop a markdown file in <code>content/notes/</code> to get started.</p>;
  }

  return (
    <div className='grid grid-cols-1 items-start gap-6 sm:grid-cols-2 lg:grid-cols-3'>
      {groups.map((group) => (
        <section key={group.category ?? '__root__'} className='notes-box'>
          <header className='notes-box-header'>
            <h2 className='notes-box-title'>{group.displayName}</h2>
          </header>
          <ul className='notes-box-list'>
            {group.posts.map((post) => (
              <li key={post.slug}>
                <Link href={`/notes/${post.slug}`} className='notes-box-link'>
                  {post.title}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
};

export default CategoryGrid;
