import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Fix Turbopack warning about multiple lockfiles
  turbopack: {
    root: __dirname,
  },
  async rewrites() {
    return [
      {
        source: '/static/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8001'}/static/:path*`,
      },
    ];
  },
};

export default nextConfig;
