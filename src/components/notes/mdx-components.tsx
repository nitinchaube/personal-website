import type { ComponentProps } from 'react';
import { stripOrderPrefixPath } from '../../lib/notes-shared';

type ImgProps = ComponentProps<'img'>;

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
  const prefix = postDir ? `/notes-assets/${postDir}/` : '/notes-assets/';

  const rewriteImageSrc = (src?: string): string | undefined => {
    if (!src) return src;
    if (/^(https?:|data:|\/)/.test(src)) return src;
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

  return { img: Img };
}
