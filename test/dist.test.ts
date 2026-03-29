import { describe, it, expect, beforeAll } from 'vitest'
import { execSync } from 'child_process'

// These tests verify the actual compiled dist/ output works correctly.
// They run after build to catch packaging issues that source-level tests miss.

beforeAll(() => {
  execSync('npm run build', { stdio: 'ignore' })
})

describe('[Dist] CJS require compatibility', () => {
  it('require() returns a callable factory function', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const bayes = require('../dist/index.cjs')
    expect(typeof bayes).toBe('function')
  })

  it('factory creates a working classifier', () => {
    const bayes = require('../dist/index.cjs')
    const c = bayes()
    expect(typeof c.learn).toBe('function')
    expect(typeof c.categorize).toBe('function')
    expect(typeof c.toJson).toBe('function')
  })

  it('fromJson is attached to factory', () => {
    const bayes = require('../dist/index.cjs')
    expect(typeof bayes.fromJson).toBe('function')
  })

  it('STATE_KEYS is attached to factory', () => {
    const bayes = require('../dist/index.cjs')
    expect(Array.isArray(bayes.STATE_KEYS)).toBe(true)
    expect(bayes.STATE_KEYS.length).toBe(8)
  })

  it('full learn → categorize cycle works', () => {
    const bayes = require('../dist/index.cjs')
    const c = bayes()
    c.learn('happy fun great', 'positive')
    c.learn('sad bad terrible', 'negative')
    const result = c.categorize('happy fun')
    expect(result.predictedCategory).toBe('positive')
    expect(result.likelihoods.length).toBe(2)
  })

  it('serialization round-trip works', () => {
    const bayes = require('../dist/index.cjs')
    const c = bayes()
    c.learn('hello world', 'greetings')
    const json = c.toJson()
    const restored = bayes.fromJson(json)
    expect(restored.categorize('hello').predictedCategory).toBe('greetings')
  })
})

describe('[Dist] ESM import compatibility', () => {
  it('default import returns a callable factory', async () => {
    const mod = await import('../dist/index.js')
    expect(typeof mod.default).toBe('function')
  })

  it('named exports are available', async () => {
    const mod = await import('../dist/index.js')
    expect(typeof mod.Naivebayes).toBe('function')
    expect(typeof mod.fromJson).toBe('function')
    expect(Array.isArray(mod.STATE_KEYS)).toBe(true)
  })

  it('factory creates a working classifier via ESM', async () => {
    const mod = await import('../dist/index.js')
    const bayes = mod.default
    const c = bayes()
    c.learn('good great', 'positive')
    c.learn('bad awful', 'negative')
    expect(c.categorize('good').predictedCategory).toBe('positive')
  })

  it('Naivebayes class can be instantiated directly', async () => {
    const { Naivebayes } = await import('../dist/index.js')
    const c = new Naivebayes()
    c.learn('hello', 'greetings')
    expect(c.categorize('hello').predictedCategory).toBe('greetings')
  })
})

describe('[Dist] Type declarations exist', () => {
  it('d.ts files are generated', () => {
    const fs = require('fs')
    expect(fs.existsSync('dist/index.d.ts')).toBe(true)
    expect(fs.existsSync('dist/index.d.cts')).toBe(true)
  })

  it('d.ts exports expected types', () => {
    const fs = require('fs')
    const dts = fs.readFileSync('dist/index.d.ts', 'utf8')
    expect(dts).toContain('NaivebayesOptions')
    expect(dts).toContain('CategorizeResult')
    expect(dts).toContain('Likelihood')
    expect(dts).toContain('InfluentialToken')
    expect(dts).toContain('CategoryStats')
    expect(dts).toContain('BatchItem')
    expect(dts).toContain('Naivebayes')
    expect(dts).toContain('fromJson')
    expect(dts).toContain('STATE_KEYS')
  })
})
