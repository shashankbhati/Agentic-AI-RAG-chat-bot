import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    lib: {
      entry: 'src/index.js',
      name: 'RagWidget',
      fileName: 'rag-widget',
      formats: ['iife'],   // single self-executing file
    },
    rollupOptions: {
      output: { inlineDynamicImports: true },
    },
    minify: true,
    outDir: 'dist',
  },
})
