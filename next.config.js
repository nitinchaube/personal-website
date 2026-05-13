/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === 'production';
// Subpath prefix when hosted at user.github.io/<repo>. Empty on custom domain
// or local dev. Set by the CI workflow from actions/configure-pages output.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

const nextConfig = {
  images: { unoptimized: true },
  // Only enable static export at build time. `next dev` is much less strict
  // (and HMR plays well with dynamic routes) when this isn't set.
  ...(isProd ? { output: 'export' } : {}),
  ...(basePath ? { basePath, assetPrefix: basePath } : {}),
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  webpack: (config) => {
    // Prevent Watchpack from trying to watch huge export/build outputs.
    config.watchOptions = {
      ...(config.watchOptions || {}),
      ignored: ['**/out/**', '**/.next/**', '**/node_modules/**'],
    };
    return config;
  },
};

module.exports = nextConfig;