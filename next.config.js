/** @type {import('next').NextConfig} */
isProd = true
const nextConfig = { 
    images: { unoptimized: true }, 
    output: 'export',
    basePath: isProd ? '/personal-website' : '',
    assetPrefix: isProd ? '/personal-website/' : '',
    eslint: {
        // Warning: This allows production builds to successfully complete even if
        // your project has ESLint errors.
        ignoreDuringBuilds: true,
      }
};

module.exports = nextConfig;
