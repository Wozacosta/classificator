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
  onSuccess: async () => {
    // Patch CJS output so `require('classificator')` returns the factory function directly
    const fs = await import('fs')
    const cjs = fs.readFileSync('dist/index.cjs', 'utf8')
    const patched = cjs + '\nif (module.exports.default) { Object.assign(module.exports.default, module.exports); module.exports = module.exports.default; }\n'
    fs.writeFileSync('dist/index.cjs', patched)
  },
})
