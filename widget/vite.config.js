import { defineConfig } from 'vite';

export default defineConfig({
  base: '/MaskGXT/',
  publicDir: 'static',
  build: {
    outDir: '../docs',
    emptyOutDir: true,
  },
});
