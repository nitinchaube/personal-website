/** @type {import('next').NextConfig} */
const nextConfig = { 
  images: { unoptimized: true }, 
  output: 'export',
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