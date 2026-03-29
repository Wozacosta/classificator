import Decimal from 'decimal.js'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Options for configuring the Naive Bayes classifier. */
export interface NaivebayesOptions {
  /** Custom tokenization function. Receives text, must return an array of tokens. */
  tokenizer?: (text: string) => string[]
  /** Transform tokens after tokenization (e.g. stopword removal, stemming). */
  tokenPreprocessor?: (tokens: string[]) => string[]
  /** Additive (Laplace) smoothing parameter. Default: 1. */
  alpha?: number
  /** Whether to use learned prior probabilities. Default: true. */
  fitPrior?: boolean
}

/** A single category likelihood result. */
export interface Likelihood {
  category: string
  logLikelihood: number
  logProba: number
  proba: number
}

/** The result of a categorize() call. */
export interface CategorizeResult {
  likelihoods: Likelihood[]
  predictedCategory: string | null
}

/** An influential token result. */
export interface InfluentialToken {
  token: string
  probability: number
  frequency: number
}

/** Per-category statistics. */
export interface CategoryStats {
  docCount: number
  wordCount: number
  vocabularySize: number
}

/** Result of getCategoryStats(). */
export interface CategoryStatsResult {
  [category: string]: CategoryStats
}

/** A batch learning item. */
export interface BatchItem {
  text: string
  category: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Keys used to serialize a classifier's state. */
export const STATE_KEYS = [
  'categories',
  'docCount',
  'totalDocuments',
  'vocabulary',
  'vocabularySize',
  'wordCount',
  'wordFrequencyCount',
  'options',
] as const

const DEFAULT_ALPHA = 1
const DEFAULT_FIT_PRIOR = true

// ---------------------------------------------------------------------------
// Default tokenizer
// ---------------------------------------------------------------------------

const defaultTokenizer = (text: string): string[] => {
  const rgxPunctuation = /[^(a-zA-ZA-Яa-я0-9_)+\s]/g
  const sanitized = text.replace(rgxPunctuation, ' ')
  return sanitized.split(/\s+/).filter(token => token.length > 0)
}

// ---------------------------------------------------------------------------
// Naivebayes class
// ---------------------------------------------------------------------------

export class Naivebayes {
  options: NaivebayesOptions
  tokenizer: (text: string) => string[]
  tokenPreprocessor: ((tokens: string[]) => string[]) | null
  alpha: number
  fitPrior: boolean
  vocabulary: Record<string, number>
  vocabularySize: number
  totalDocuments: number
  docCount: Record<string, number>
  wordCount: Record<string, number>
  wordFrequencyCount: Record<string, Record<string, number>>
  categories: Record<string, boolean>

  constructor(options?: NaivebayesOptions) {
    this.options = {}
    if (typeof options !== 'undefined') {
      if (!options || typeof options !== 'object' || Array.isArray(options)) {
        throw TypeError(
          `NaiveBayes got invalid 'options': '${options}'. Pass in an object.`
        )
      }
      this.options = options
    }

    if (this.options.tokenizer && typeof this.options.tokenizer !== 'function') {
      throw TypeError('NaiveBayes: tokenizer must be a function.')
    }
    if (this.options.tokenPreprocessor && typeof this.options.tokenPreprocessor !== 'function') {
      throw TypeError('NaiveBayes: tokenPreprocessor must be a function.')
    }

    this.tokenizer = this.options.tokenizer || defaultTokenizer
    this.tokenPreprocessor = this.options.tokenPreprocessor || null
    this.alpha = this.options.alpha === undefined ? DEFAULT_ALPHA : this.options.alpha
    this.fitPrior = this.options.fitPrior === undefined ? DEFAULT_FIT_PRIOR : this.options.fitPrior

    this.vocabulary = {}
    this.vocabularySize = 0
    this.totalDocuments = 0
    this.docCount = {}
    this.wordCount = {}
    this.wordFrequencyCount = {}
    this.categories = {}
  }

  /** Tokenize text and optionally apply the preprocessor. */
  tokenize(text: string): string[] {
    const tokens = this.tokenizer(text)
    if (this.tokenPreprocessor) {
      return this.tokenPreprocessor(tokens)
    }
    return tokens
  }

  /** Initialize data structure entries for a new category. */
  initializeCategory(categoryName: string): this {
    if (!this.categories[categoryName]) {
      this.docCount[categoryName] = 0
      this.wordCount[categoryName] = 0
      this.wordFrequencyCount[categoryName] = {}
      this.categories[categoryName] = true
    }
    return this
  }

  /** Remove a category and all its associated data. */
  removeCategory(categoryName: string): this {
    if (!this.categories[categoryName]) {
      return this
    }
    this.totalDocuments -= this.docCount[categoryName]

    Object.keys(this.wordFrequencyCount[categoryName]).forEach((token) => {
      if (this.vocabulary[token] && this.vocabulary[token] > 0) {
        this.vocabulary[token]--
        if (this.vocabulary[token] === 0) this.vocabularySize--
      }
    })

    delete this.docCount[categoryName]
    delete this.wordCount[categoryName]
    delete this.wordFrequencyCount[categoryName]
    delete this.categories[categoryName]

    return this
  }

  /** Train the classifier: associate `text` with `category`. */
  learn(text: string, category: string): this {
    if (typeof text !== 'string') {
      throw new TypeError(`NaiveBayes: text must be a string, got ${typeof text}.`)
    }
    if (typeof category !== 'string') {
      throw new TypeError(`NaiveBayes: category must be a string, got ${typeof category}.`)
    }

    this.initializeCategory(category)

    this.docCount[category]++
    this.totalDocuments++

    const tokens = this.tokenize(text)
    const freqTable = this.frequencyTable(tokens)

    Object.keys(freqTable).forEach((token) => {
      const frequencyInText = freqTable[token]

      if (!this.vocabulary[token] || this.vocabulary[token] === 0) {
        this.vocabularySize++
        this.vocabulary[token] = 1
      } else {
        this.vocabulary[token]++
      }

      if (!this.wordFrequencyCount[category][token]) {
        this.wordFrequencyCount[category][token] = frequencyInText
      } else {
        this.wordFrequencyCount[category][token] += frequencyInText
      }

      this.wordCount[category] += frequencyInText
    })

    return this
  }

  /** Untrain the classifier: remove association of `text` with `category`. */
  unlearn(text: string, category: string): this {
    if (typeof text !== 'string') {
      throw new TypeError(`NaiveBayes: text must be a string, got ${typeof text}.`)
    }
    if (typeof category !== 'string') {
      throw new TypeError(`NaiveBayes: category must be a string, got ${typeof category}.`)
    }
    if (!this.categories[category]) {
      throw new Error(`NaiveBayes: cannot unlearn from non-existent category: '${category}'.`)
    }

    this.docCount[category]--
    if (this.docCount[category] === 0) {
      delete this.docCount[category]
    }

    this.totalDocuments--

    const tokens = this.tokenize(text)
    const freqTable = this.frequencyTable(tokens)

    Object.keys(freqTable).forEach((token) => {
      const frequencyInText = freqTable[token]

      if (this.vocabulary[token] && this.vocabulary[token] > 0) {
        this.vocabulary[token]--
        if (this.vocabulary[token] === 0) this.vocabularySize--
      }

      if (this.wordFrequencyCount[category] && this.wordFrequencyCount[category][token]) {
        this.wordFrequencyCount[category][token] -= frequencyInText
        if (this.wordFrequencyCount[category][token] <= 0) {
          delete this.wordFrequencyCount[category][token]
        }
      }

      if (this.wordCount[category] !== undefined) {
        this.wordCount[category] -= frequencyInText
        if (this.wordCount[category] <= 0) {
          delete this.wordCount[category]
          delete this.wordFrequencyCount[category]
        }
      }
    })

    if (!this.docCount[category]) {
      delete this.categories[category]
    }

    return this
  }

  /** Determine what category `text` belongs to. */
  categorize(text: string): CategorizeResult {
    if (typeof text !== 'string') {
      throw new TypeError(`NaiveBayes: text must be a string, got ${typeof text}.`)
    }

    const tokens = this.tokenize(text)
    const freqTable = this.frequencyTable(tokens)
    const categoryNames = Object.keys(this.categories)

    if (categoryNames.length === 0) {
      return { likelihoods: [], predictedCategory: null }
    }

    interface InternalLikelihood {
      category: string
      logLikelihood: Decimal
    }

    const likelihoods: InternalLikelihood[] = []

    categoryNames.forEach((category) => {
      let categoryLikelihood: number
      if (this.fitPrior) {
        categoryLikelihood = this.docCount[category] / this.totalDocuments
      } else {
        categoryLikelihood = 1
      }

      let logLikelihood = new Decimal(categoryLikelihood).naturalLogarithm()

      Object.keys(freqTable).forEach((token) => {
        if (this.vocabulary[token] && this.vocabulary[token] > 0) {
          const termFrequencyInText = freqTable[token]
          const prob = this.tokenProbability(token, category)

          const logTokenProb = new Decimal(prob).naturalLogarithm()
          logLikelihood = logLikelihood.plus(logTokenProb.times(termFrequencyInText))
        }
      })

      likelihoods.push({ category, logLikelihood })
    })

    // Numerically stable logsumexp: subtract max to prevent overflow/underflow
    const logsumexp = (items: InternalLikelihood[]): Decimal => {
      if (items.length === 0) return new Decimal(0)

      const maxLog = items.reduce((max, l) => {
        return l.logLikelihood.greaterThan(max) ? l.logLikelihood : max
      }, items[0].logLikelihood)

      let sum = new Decimal(0)
      items.forEach((item) => {
        const shifted = item.logLikelihood.minus(maxLog)
        sum = sum.plus(Decimal.exp(shifted))
      })

      return maxLog.plus(sum.naturalLogarithm())
    }

    const logProbX = logsumexp(likelihoods)

    const result: Likelihood[] = likelihoods.map((l) => {
      const logProba = l.logLikelihood.minus(logProbX)
      const proba = logProba.naturalExponential()
      return {
        category: l.category,
        logLikelihood: l.logLikelihood.toNumber(),
        logProba: logProba.toNumber(),
        proba: proba.toNumber(),
      }
    })

    result.sort((a, b) => b.proba - a.proba)

    return {
      likelihoods: result,
      predictedCategory: result[0].category,
    }
  }

  /** Like categorize(), but returns only the top N most likely categories. */
  categorizeTopN(text: string, n: number): CategorizeResult {
    const result = this.categorize(text)
    if (result.likelihoods.length > n) {
      result.likelihoods = result.likelihoods.slice(0, n)
    }
    return result
  }

  /** Categorize with a confidence threshold. Returns null predictedCategory if below threshold. */
  categorizeWithConfidence(text: string, threshold: number): CategorizeResult {
    if (typeof threshold !== 'number' || threshold < 0 || threshold > 1) {
      throw new TypeError('NaiveBayes: threshold must be a number between 0 and 1.')
    }
    const result = this.categorize(text)
    if (result.predictedCategory === null) return result

    if (result.likelihoods[0].proba < threshold) {
      result.predictedCategory = null
    }
    return result
  }

  /** Get the top N most influential tokens for a text's classification. */
  topInfluentialTokens(text: string, n?: number): InfluentialToken[] {
    const limit = (n === undefined || n === null) ? 5 : Math.max(0, Math.floor(n))
    const tokens = this.tokenize(text)
    const freqTable = this.frequencyTable(tokens)
    const result = this.categorize(text)
    const topCategory = result.predictedCategory

    if (!topCategory) return []

    return Object.keys(freqTable)
      .filter(token => this.vocabulary[token] && this.vocabulary[token] > 0)
      .map(token => ({
        token,
        probability: this.tokenProbability(token, topCategory),
        frequency: freqTable[token],
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, limit)
  }

  /** Calculate probability that a token belongs to a category. */
  tokenProbability(token: string, category: string): number {
    const wordFreqCount = this.wordFrequencyCount[category][token] || 0
    const wc = this.wordCount[category]
    return (wordFreqCount + this.alpha) / (wc + this.alpha * this.vocabularySize)
  }

  /** Build a frequency hashmap from an array of tokens. */
  frequencyTable(tokens: string[]): Record<string, number> {
    const table: Record<string, number> = Object.create(null)
    tokens.forEach((token) => {
      if (!table[token]) table[token] = 1
      else table[token]++
    })
    return table
  }

  /** Serialize the classifier's state as a JSON string. */
  toJson(): string {
    const state: Record<string, unknown> = {}
    STATE_KEYS.forEach(k => {
      state[k] = (this as unknown as Record<string, unknown>)[k]
    })
    return JSON.stringify(state)
  }

  /** Get an array of all category names the classifier has learned. */
  getCategories(): string[] {
    return Object.keys(this.categories)
  }

  /** Learn from multiple text/category pairs at once. */
  learnBatch(items: BatchItem[]): this {
    if (!Array.isArray(items)) {
      throw new TypeError('NaiveBayes: learnBatch expects an array of { text, category } objects.')
    }
    items.forEach(item => {
      this.learn(item.text, item.category)
    })
    return this
  }

  /** Reset the classifier to its initial untrained state, preserving options. */
  reset(): this {
    this.vocabulary = {}
    this.vocabularySize = 0
    this.totalDocuments = 0
    this.docCount = {}
    this.wordCount = {}
    this.wordFrequencyCount = {}
    this.categories = {}
    return this
  }

  /** Get statistics about each category's training data. */
  getCategoryStats(): CategoryStatsResult {
    const stats: Record<string, CategoryStats> = {}
    Object.keys(this.categories).forEach(category => {
      stats[category] = {
        docCount: this.docCount[category] || 0,
        wordCount: this.wordCount[category] || 0,
        vocabularySize: Object.keys(this.wordFrequencyCount[category] || {}).length,
      }
    })
    const totalWordCount = Object.keys(this.categories).reduce((sum, cat) => {
      return sum + (this.wordCount[cat] || 0)
    }, 0)
    stats._total = {
      docCount: this.totalDocuments,
      wordCount: totalWordCount,
      vocabularySize: this.vocabularySize,
    }
    return stats as CategoryStatsResult
  }
}

// ---------------------------------------------------------------------------
// Static: fromJson
// ---------------------------------------------------------------------------

/** Restore a classifier from its JSON representation. */
export function fromJson(jsonStrOrObject: string | object, options?: NaivebayesOptions): Naivebayes {
  let parameters: Record<string, unknown>

  try {
    switch (typeof jsonStrOrObject) {
      case 'string':
        parameters = JSON.parse(jsonStrOrObject)
        break

      case 'object':
        if (jsonStrOrObject === null) {
          throw new Error('')
        }
        parameters = jsonStrOrObject as Record<string, unknown>
        break

      default:
        throw new Error('')
    }
  } catch {
    throw new Error('NaiveBayes.fromJson expects a valid JSON string or an object.')
  }

  const restoredOptions = Object.assign(
    {},
    parameters.options as NaivebayesOptions | undefined,
    options
  )

  const classifier = new Naivebayes(restoredOptions)

  STATE_KEYS.forEach((k) => {
    if (typeof parameters[k] === 'undefined') {
      throw new Error(
        `NaiveBayes.fromJson: JSON string is missing an expected property: [${k}].`
      )
    }
    ;(classifier as unknown as Record<string, unknown>)[k] = parameters[k]
  })

  // Restore merged options (STATE_KEYS includes 'options' which overwrites
  // with saved state, losing runtime-only options like tokenizer/tokenPreprocessor)
  classifier.options = restoredOptions

  return classifier
}

// ---------------------------------------------------------------------------
// Factory function — preserves backward compat for both CJS and ESM
// ---------------------------------------------------------------------------

export interface ClassifierFactory {
  (options?: NaivebayesOptions): Naivebayes
  fromJson: typeof fromJson
  STATE_KEYS: typeof STATE_KEYS
  Naivebayes: typeof Naivebayes
}

const bayes = function createClassifier(options?: NaivebayesOptions): Naivebayes {
  return new Naivebayes(options)
} as ClassifierFactory

bayes.fromJson = fromJson
bayes.STATE_KEYS = STATE_KEYS
bayes.Naivebayes = Naivebayes

export default bayes

// CJS compat: when bundled to CJS, module.exports = bayes
// This is handled by tsup's cjsInterop option which rewrites
// module.exports = module.exports.default when a default export exists
