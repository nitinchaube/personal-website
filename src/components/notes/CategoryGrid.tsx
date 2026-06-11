import type { NoteTreeNode } from '../../lib/notes-shared';
import NoteTree from './NoteTree';

type Props = {
  nodes: NoteTreeNode[];
};

const CategoryGrid = ({ nodes }: Props) => {
  if (nodes.length === 0) {
    return (
      <p className='notes-meta'>
        No notes yet. Drop a markdown file in <code>content/notes/</code> to get started.
      </p>
    );
  }

  const topFolders = nodes.filter((n) => n.type === 'folder');
  const loosePosts = nodes.filter((n) => n.type === 'post');

  type Box = { key: string; displayName: string; children: NoteTreeNode[] };
  const boxes: Box[] = topFolders.map((folder) => ({
    key: `folder:${folder.type === 'folder' ? folder.name : ''}`,
    displayName: folder.type === 'folder' ? folder.displayName : '',
    children: folder.type === 'folder' ? folder.children : [],
  }));
  if (loosePosts.length > 0) {
    boxes.push({ key: '__general__', displayName: 'General', children: loosePosts });
  }

  return (
    <div className='columns-1 gap-6 sm:columns-2 lg:columns-3'>
      {boxes.map((box) => (
        <section key={box.key} className='notes-box mb-6 break-inside-avoid'>
          <header className='notes-box-header'>
            <h2 className='notes-box-title'>{box.displayName}</h2>
          </header>
          <NoteTree nodes={box.children} />
        </section>
      ))}
    </div>
  );
};

export default CategoryGrid;
