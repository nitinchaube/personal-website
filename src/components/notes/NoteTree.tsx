import Link from 'next/link';
import type { NoteTreeNode } from '../../lib/notes-shared';

type Props = {
  nodes: NoteTreeNode[];
  /** Nesting depth, used only to control default-open behavior. */
  level?: number;
};

/**
 * Renders a nested folder tree. Folders are collapsible via native
 * <details>/<summary> (no client JS, so it works in the static export);
 * posts are plain links. Subfolders start collapsed; click to expand.
 */
const NoteTree = ({ nodes, level = 0 }: Props) => {
  if (nodes.length === 0) return null;

  return (
    <ul className='notes-tree' data-level={level}>
      {nodes.map((node) => {
        if (node.type === 'post') {
          return (
            <li key={node.slug} className='notes-tree-item'>
              <Link href={`/notes/${node.slug}`} className='notes-box-link'>
                {node.title}
              </Link>
            </li>
          );
        }

        return (
          <li key={`folder:${node.name}`} className='notes-tree-item'>
            <details className='notes-tree-folder'>
              <summary className='notes-tree-summary'>
                <span className='notes-tree-folder-name'>{node.displayName}</span>
              </summary>
              <div className='notes-tree-children'>
                <NoteTree nodes={node.children} level={level + 1} />
              </div>
            </details>
          </li>
        );
      })}
    </ul>
  );
};

export default NoteTree;
