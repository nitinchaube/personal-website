/**
 * Base path the site is served from (e.g. "/personal-website" on
 * GitHub Pages project sites; empty string on custom-domain Pages
 * or local dev).
 *
 * Set at build time via the NEXT_PUBLIC_BASE_PATH env var, which the
 * GitHub Actions workflow derives from `actions/configure-pages` output.
 */
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

/**
 * Prepend BASE_PATH to an absolute path so it resolves correctly when the
 * site is hosted at a subpath. Pass-through for external / relative / hash
 * URLs. Idempotent (won't double-prefix).
 *
 * Use this for raw `<img src=...>`, `<link href=...>`, `window.location.href`,
 * etc. Next.js's <Link> and <Image> handle basePath automatically — don't
 * apply withBase() to those.
 */
export function withBase(p: string): string {
  if (!p) return p;
  if (/^(https?:|data:|mailto:|tel:|#)/.test(p)) return p;
  if (!p.startsWith('/')) return p;
  if (!BASE_PATH) return p;
  if (p === BASE_PATH || p.startsWith(BASE_PATH + '/')) return p;
  return BASE_PATH + p;
}
