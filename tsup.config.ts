import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  target: 'node18',
  outDir: 'dist',
  footer: {
    // Make `require('classificator')` return the factory function directly
    js: 'if (module.exports.default) { Object.assign(module.exports.default, module.exports); module.exports = module.exports.default; }',
  },
})
