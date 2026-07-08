import type { ComponentProps } from 'react';
import { stripOrderPrefixPath } from '../../lib/notes-shared';
import { withBase } from '../../lib/site';

type ImgProps = ComponentProps<'img'>;
type AnchorProps = ComponentProps<'a'>;

/**
 * Build an MDX component map for a specific post.
 *
 * `postDir` is the post's directory relative to `content/notes/` (e.g. "fundamentals" or ""),
 * already with any numeric ordering prefix stripped.
 *
 * Cursor's auto-paste inserts image paths like `assets/<post-basename>/image-1.png` relative
 * to the .md file. Our sync script mirrors these to `public/notes-assets/<postDir>/<rest>`
 * with prefixes stripped from every directory segment, so the rewrite here strips them too.
 */
export function mdxComponentsFor(postDir: string) {
  const prefix = withBase(postDir ? `/notes-assets/${postDir}/` : '/notes-assets/');

  const rewriteImageSrc = (src?: string): string | undefined => {
    if (!src) return src;
    if (/^(https?:|data:)/.test(src)) return src;
    if (src.startsWith('/')) return withBase(src);
    // Strip leading "./" and a possible "assets/" prefix from the relative link.
    const cleaned = src.replace(/^\.\//, '').replace(/^assets\//, '');
    // Strip any "01-" / "1_" prefix from each remaining segment so the URL
    // matches what the sync script writes to public/notes-assets.
    return prefix + stripOrderPrefixPath(cleaned);
  };

  const Img = ({ src, alt, ...rest }: ImgProps) => {
    const finalSrc = typeof src === 'string' ? rewriteImageSrc(src) : src;
    if (alt && alt.trim().length > 0) {
      return (
        <figure>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={finalSrc} alt={alt} loading='lazy' {...rest} />
          <figcaption>{alt}</figcaption>
        </figure>
      );
    }
    // eslint-disable-next-line @next/next/no-img-element
    return <img src={finalSrc} alt='' loading='lazy' {...rest} />;
  };

  // Note content links to other notes with site-absolute paths like
  // `/notes/Category/Slug`. Plain markdown renders these as bare <a> tags
  // (not Next's <Link>), so they need the same basePath rewrite as images
  // to resolve correctly on GitHub Pages project sites (e.g. /personal-website).
  const Anchor = ({ href, ...rest }: AnchorProps) => {
    const finalHref = typeof href === 'string' ? withBase(href) : href;
    return <a href={finalHref} {...rest} />;
  };

  return { img: Img, a: Anchor };
}
