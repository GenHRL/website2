/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    unoptimized: true, // Or false if deploying to Vercel/similar
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },
  // If you plan to use next/image for optimization, you might need a custom loader for GitHub Pages.
  // For simplicity, we'll assume standard img tags or manually handling image paths for now.
  // images: {
  //   loader: 'custom',
  //   loaderFile: './image-loader.js', // You would need to create this file
  // },
  // Trailing slash can be important for how paths are resolved in static exports.
  // It ensures that /path becomes /path/index.html
  // trailingSlash: true,
  // Add other Next.js configurations here as needed
};

module.exports = nextConfig; 