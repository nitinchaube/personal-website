import { compileMDX } from 'next-mdx-remote/rsc';
import { notFound } from 'next/navigation';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypePrettyCode from 'rehype-pretty-code';
import rehypeKatex from 'rehype-katex';

import { getAdjacentNotes, getAllSlugSegments, getNoteBySlug } from '../../../lib/notes';
import PostHeader from '../../../components/notes/PostHeader';
import PrevNext from '../../../components/notes/PrevNext';
import Toc from '../../../components/notes/Toc';
import MobileToc from '../../../components/notes/MobileToc';
import { mdxComponentsFor } from '../../../components/notes/mdx-components';

export async function generateStaticParams() {
  return getAllSlugSegments().map((segments) => ({ slug: segments }));
}

export const dynamic = 'force-static';
export const dynamicParams = false;

/** Slug segments from the router may be percent-encoded (e.g. spaces in folder names). */
function slugFromParams(segments: string[]): string {
  return segments.map((s) => decodeURIComponent(s)).join('/');
}

export function generateMetadata({ params }: { params: { slug: string[] } }) {
  const slug = slugFromParams(params.slug);
  const note = getNoteBySlug(slug);
  if (!note) return { title: "Nitin's Desk" };
  return {
    title: `${note.title} | Nitin's Desk`,
    description: note.summary ?? undefined,
  };
}

export default async function NotePage({ params }: { params: { slug: string[] } }) {
  const slug = slugFromParams(params.slug);
  const note = getNoteBySlug(slug);
  if (!note) notFound();
  const { prev, next } = getAdjacentNotes(slug);

  const { content } = await compileMDX({
    source: note.content,
    components: mdxComponentsFor(note.postDir),
    options: {
      mdxOptions: {
        // `.md` notes use plain markdown so `{...}` in LaTeX is not parsed as JSX.
        format: note.contentFormat,
        remarkPlugins: [remarkGfm, remarkMath],
        rehypePlugins: [
          rehypeSlug,
          [
            rehypeAutolinkHeadings,
            {
              behavior: 'append',
              properties: { className: ['heading-anchor'], ariaHidden: true, tabIndex: -1 },
              content: { type: 'text', value: '#' },
            },
          ],
          [rehypePrettyCode, { theme: 'github-dark', keepBackground: true }],
          rehypeKatex,
        ],
      },
    },
  });

  return (
    <main className='mx-auto w-full max-w-[1400px] px-6 py-14 sm:px-10 sm:py-20'>
      <div className='lg:grid lg:grid-cols-[minmax(0,1fr)_200px] lg:gap-24 xl:gap-32'>
        <article className='notes-article mx-auto w-full max-w-[860px]'>
          <PostHeader note={note} />
          <div className='prose prose-notes max-w-none'>{content}</div>
          <PrevNext prev={prev} next={next} />
        </article>
        <aside className='hidden lg:block lg:justify-self-end'>
          <div className='sticky top-24'>
            <Toc selector='.notes-article' />
          </div>
        </aside>
      </div>
      <MobileToc selector='.notes-article' />
    </main>
  );
}
