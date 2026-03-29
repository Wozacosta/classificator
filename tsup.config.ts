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
    // Patch CJS output so `require('classificator')` returns the factory function directly.
    // Insert before the sourcemap comment so debuggers can still find source maps.
    const fs = await import('fs')
    const cjs = fs.readFileSync('dist/index.cjs', 'utf8')
    const patch = '\nif (module.exports.default) { Object.assign(module.exports.default, module.exports); module.exports = module.exports.default; }\n'
    const sourcemapMatch = cjs.match(/\n\/\/# sourceMappingURL=.*$/)
    const patched = sourcemapMatch
      ? cjs.replace(sourcemapMatch[0], patch + sourcemapMatch[0])
      : cjs + patch
    fs.writeFileSync('dist/index.cjs', patched)
  },
})
