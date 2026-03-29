import { describe, it, expect } from 'vitest'
import bayes from '../src/index'

describe('bayes() init', () => {
  it('valid options (falsey or with an object) do not raise Errors', () => {
    const validOptionsCases = [ undefined, {} ];

    validOptionsCases.forEach((validOptions) => {
      const classifier = bayes(validOptions)
      expect(classifier.options).toEqual({})
    })
  })

  it('invalid options (truthy and not object) raise TypeError during init', () => {
    const invalidOptionsCases = [ null, 0, 'a', [] ];

    invalidOptionsCases.forEach((invalidOptions) => {
      expect(() => { bayes(invalidOptions) }).toThrow(Error)
      expect(() => { bayes(invalidOptions) }).toThrow(TypeError)
    })
  })

  it('throws TypeError when tokenizer is not a function', () => {
    expect(() => { bayes({ tokenizer: 'not a function' }) }).toThrow(TypeError)
    expect(() => { bayes({ tokenizer: 42 }) }).toThrow(TypeError)
  })

  it('throws TypeError when tokenPreprocessor is not a function', () => {
    expect(() => { bayes({ tokenPreprocessor: 'bad' }) }).toThrow(TypeError)
  })
})

describe('bayes using custom tokenizer', () => {
  it('uses custom tokenization function if one is provided in `options`.', () => {
    const splitOnChar = (text) => {
      return text.split('')
    }

    const classifier = bayes({ tokenizer: splitOnChar })

    classifier.learn('abcd', 'happy')

    expect(classifier.totalDocuments).toBe(1)
    expect(classifier.docCount.happy).toBe(1)
    expect(classifier.vocabulary).toEqual({ a: 1, b: 1, c: 1, d: 1 })
    expect(classifier.vocabularySize).toBe(4)
    expect(classifier.wordCount.happy).toBe(4)
    expect(classifier.wordFrequencyCount.happy.a).toBe(1)
    expect(classifier.wordFrequencyCount.happy.b).toBe(1)
    expect(classifier.wordFrequencyCount.happy.c).toBe(1)
    expect(classifier.wordFrequencyCount.happy.d).toBe(1)
    expect(classifier.categories).toStrictEqual({ happy: true })
  })
})

describe('bayes using tokenPreprocessor', () => {
  it('applies tokenPreprocessor after tokenizer', () => {
    const stopwords = new Set(['the', 'a', 'is', 'in'])
    const classifier = bayes({
      tokenPreprocessor: (tokens) => {
        return tokens
          .map((t) => { return t.toLowerCase() })
          .filter((t) => { return !stopwords.has(t) })
      }
    })

    classifier.learn('The cat is in a hat', 'animals')

    // stopwords should be removed
    expect(classifier.wordFrequencyCount.animals['the']).toBe(undefined)
    expect(classifier.wordFrequencyCount.animals['a']).toBe(undefined)
    expect(classifier.wordFrequencyCount.animals['is']).toBe(undefined)
    expect(classifier.wordFrequencyCount.animals['in']).toBe(undefined)

    // content words should remain (lowercased)
    expect(classifier.wordFrequencyCount.animals['cat']).toBe(1)
    expect(classifier.wordFrequencyCount.animals['hat']).toBe(1)
  })

  it('works with stemming-style preprocessor', () => {
    const classifier = bayes({
      tokenPreprocessor: (tokens) => {
        return tokens.map((t) => {
          // crude stemming: strip trailing 's', 'ing', 'ed'
          return t.replace(/(ing|ed|s)$/i, '').toLowerCase()
        })
      }
    })

    classifier.learn('running dogs played', 'active')
    classifier.learn('sleeping cats rested', 'passive')

    const result = classifier.categorize('dogs playing')
    expect(result.predictedCategory).toBe('active')
  })

  it('preprocessor is preserved through fromJson with options', () => {
    const preprocessor = (tokens) => {
      return tokens.map((t) => { return t.toLowerCase() })
    }

    const classifier = bayes({ tokenPreprocessor: preprocessor })
    classifier.learn('HELLO', 'greetings')

    const revived = bayes.fromJson(classifier.toJson(), { tokenPreprocessor: preprocessor })
    const result = revived.categorize('HELLO')
    expect(result.predictedCategory).toBe('greetings')
  })

  it('classifier.options preserves runtime options after fromJson', () => {
    const preprocessor = (tokens) => {
      return tokens.map((t) => { return t.toLowerCase() })
    }
    const tokenizer = (text) => { return text.split(' ') }

    const classifier = bayes({ tokenPreprocessor: preprocessor, tokenizer: tokenizer })
    classifier.learn('HELLO WORLD', 'greetings')

    const revived = bayes.fromJson(classifier.toJson(), {
      tokenPreprocessor: preprocessor,
      tokenizer: tokenizer
    })

    expect(revived.options.tokenPreprocessor).toBe(preprocessor)
    expect(revived.options.tokenizer).toBe(tokenizer)
    expect(revived.tokenPreprocessor).toBe(preprocessor)
    expect(revived.tokenizer).toBe(tokenizer)
  })
})

describe('bayes serializing/deserializing its state', () => {
  it('serializes/deserializes its state as JSON correctly.', () => {
    const classifier = bayes()

    classifier.learn('Fun times were had by all', 'positive')
    classifier.learn('sad dark rainy day in the cave', 'negative')

    const jsonRepr = classifier.toJson()
    const state = JSON.parse(jsonRepr)

    bayes.STATE_KEYS.forEach((k) => {
      expect(state[k]).toEqual(classifier[k])
    })

    const revivedClassifier = bayes.fromJson(jsonRepr)

    bayes.STATE_KEYS.forEach((k) => {
      expect(revivedClassifier[k]).toEqual(classifier[k])
    })
  })
})

describe('bayes using custom tokenizer with fromJson', () => {
  it('accepts a custom tokenizer passed as an option to fromJson', () => {
    const splitOnChar = (text) => {
      return text.split('')
    }

    const classifier = bayes({ tokenizer: splitOnChar })

    classifier.learn('abcd', 'happy')
    classifier.learn('efgh', 'sad')

    const jsonRepr = classifier.toJson()
    const revivedClassifier = bayes.fromJson(jsonRepr, { tokenizer: splitOnChar })

    const result = revivedClassifier.categorize('abcd')
    expect(result.predictedCategory).toBe('happy')
  })
})

describe('bayes .fromJson() edge cases', () => {
  it('throws on null input', () => {
    expect(() => { bayes.fromJson(null) }).toThrow(Error)
  })

  it('throws on numeric input', () => {
    expect(() => { bayes.fromJson(42) }).toThrow(Error)
  })

  it('throws on invalid JSON string', () => {
    expect(() => { bayes.fromJson('not valid json') }).toThrow(Error)
  })

  it('throws when JSON is missing required state keys', () => {
    expect(() => { bayes.fromJson('{"options":{}}') }).toThrow(Error)
  })

  it('preserves options through serialization round-trip', () => {
    const classifier = bayes({ alpha: 2, fitPrior: false })
    classifier.learn('hello world', 'greetings')

    const revived = bayes.fromJson(classifier.toJson())
    expect(revived.alpha).toBe(2)
    expect(revived.fitPrior).toBe(false)
  })
})

describe('bayes .learn() correctness', () => {
  it('categorizes correctly for `positive` and `negative` categories', () => {
    let classifier = bayes();

    classifier.learn('amazing, awesome movie!! Yeah!!', 'positive')
    classifier.learn('Sweet, this is incredibly, amazing, perfect, great!!', 'positive')
    classifier.learn('terrible, shitty thing. Damn. Sucks!!', 'negative')
    classifier.learn('I dont really know what to make of this.', 'neutral')

    expect(classifier.categorize('awesome, cool, amazing!! Yay.').predictedCategory).toEqual('positive')
  })

  it('categorizes correctly for `chinese` and `japanese` categories', () => {
    const classifier = bayes()

    classifier.learn('Chinese Beijing Chinese', 'chinese')
    classifier.learn('Chinese Chinese Shanghai', 'chinese')
    classifier.learn('Chinese Macao', 'chinese')
    classifier.learn('Tokyo Japan Chinese', 'japanese')

    const chineseFrequencyCount = classifier.wordFrequencyCount.chinese

    expect(chineseFrequencyCount['Chinese']).toBe(5)
    expect(chineseFrequencyCount['Beijing']).toBe(1)
    expect(chineseFrequencyCount['Shanghai']).toBe(1)
    expect(chineseFrequencyCount['Macao']).toBe(1)

    const japaneseFrequencyCount = classifier.wordFrequencyCount.japanese

    expect(japaneseFrequencyCount['Tokyo']).toBe(1)
    expect(japaneseFrequencyCount['Japan']).toBe(1)
    expect(japaneseFrequencyCount['Chinese']).toBe(1)

    expect(classifier.categorize('Chinese Chinese Chinese Tokyo Japan').predictedCategory).toEqual('chinese')
  })

  it('correctly tokenizes cyrlic characters', () => {
    const classifier = bayes()

    classifier.learn('Надежда за', 'a')
    classifier.learn('Надежда за обич еп.36 Тест', 'b')
    classifier.learn('Надежда за обич еп.36 Тест', 'b')

    const aFreqCount = classifier.wordFrequencyCount.a
    expect(aFreqCount['Надежда']).toBe(1)
    expect(aFreqCount['за']).toBe(1)

    const bFreqCount = classifier.wordFrequencyCount.b
    expect(bFreqCount['Надежда']).toBe(2)
    expect(bFreqCount['за']).toBe(2)
    expect(bFreqCount['обич']).toBe(2)
    expect(bFreqCount['еп']).toBe(2)
    expect(bFreqCount['36']).toBe(2)
    expect(bFreqCount['Тест']).toBe(2)
  })

  it('correctly computes probabilities without prior', () => {
    const classifier = bayes({ fitPrior: false})

    classifier.learn('aa', '1')
    classifier.learn('aa', '1')
    classifier.learn('aa', '1')
    classifier.learn('bb', '2')

    expect(classifier.categorize('cc').likelihoods[0].proba).toBe(0.5)
    expect(Number(classifier.categorize('bb').likelihoods[0].proba).toFixed(6)).toBe(Number(0.76923077).toFixed(6))
    expect(Number(classifier.categorize('aa').likelihoods[0].proba).toFixed(6)).toBe(Number(0.70588235).toFixed(6))
  })

  it('correctly computes probabilities with prior', () => {
    const classifier = bayes()

    classifier.learn('aa', '1')
    classifier.learn('aa', '1')
    classifier.learn('aa', '1')
    classifier.learn('bb', '2')

    expect(classifier.categorize('cc').likelihoods[0].proba).toBe(0.75)
    expect(Number(classifier.categorize('bb').likelihoods[0].proba).toFixed(6)).toBe(Number(0.52631579).toFixed(6))
    expect(Number(classifier.categorize('aa').likelihoods[0].proba).toFixed(6)).toBe(Number(0.87804878).toFixed(6))
  })

  it('throws TypeError when text is not a string', () => {
    const classifier = bayes()
    expect(() => { classifier.learn(123, 'cat') }).toThrow(TypeError)
    expect(() => { classifier.learn(null, 'cat') }).toThrow(TypeError)
  })

  it('throws TypeError when category is not a string', () => {
    const classifier = bayes()
    expect(() => { classifier.learn('hello', 123) }).toThrow(TypeError)
    expect(() => { classifier.learn('hello', null) }).toThrow(TypeError)
  })
})

describe('bayes .unlearn() correctness', () => {
  it('reverses the effect of a single learn call', () => {
    const classifier = bayes()

    classifier.learn('fun times', 'positive')
    classifier.learn('bad times', 'negative')
    classifier.learn('great day', 'positive')

    const docsBefore = classifier.totalDocuments
    classifier.unlearn('fun times', 'positive')

    expect(classifier.totalDocuments).toBe(docsBefore - 1)
  })

  it('throws when unlearning from a non-existent category', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')

    expect(() => {
      classifier.unlearn('hello', 'nonexistent')
    }).toThrow(Error)
  })

  it('removes category from categories when last doc is unlearned', () => {
    const classifier = bayes()

    classifier.learn('hello', 'greetings')
    expect(classifier.categories['greetings']).toBeTruthy()

    classifier.unlearn('hello', 'greetings')
    expect(classifier.categories['greetings']).toBe(undefined)
  })

  it('classifier still works correctly after unlearn', () => {
    const classifier = bayes()

    classifier.learn('amazing great', 'positive')
    classifier.learn('terrible awful', 'negative')
    classifier.learn('bad horrible', 'negative')

    classifier.unlearn('bad horrible', 'negative')

    const result = classifier.categorize('terrible')
    expect(result.predictedCategory).toBe('negative')
  })

  it('returns this for method chaining', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')
    const result = classifier.unlearn('hello', 'greetings')
    expect(result).toBe(classifier)
  })

  it('throws TypeError when text is not a string', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')
    expect(() => { classifier.unlearn(123, 'greetings') }).toThrow(TypeError)
  })

  it('does not leave negative wordCount', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')
    classifier.unlearn('hello', 'greetings')

    expect(classifier.wordCount['greetings']).toBe(undefined)
  })
})

describe('bayes .removeCategory()', () => {
  it('removes a category and its associated data', () => {
    const classifier = bayes()

    classifier.learn('hello world', 'greetings')
    classifier.learn('bad stuff', 'negative')

    classifier.removeCategory('greetings')

    expect(classifier.categories['greetings']).toBe(undefined)
    expect(classifier.docCount['greetings']).toBe(undefined)
    expect(classifier.wordCount['greetings']).toBe(undefined)
    expect(classifier.wordFrequencyCount['greetings']).toBe(undefined)
  })

  it('returns this for chaining', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')
    const result = classifier.removeCategory('greetings')
    expect(result).toBe(classifier)
  })

  it('returns this when removing non-existent category (no-op)', () => {
    const classifier = bayes()
    const result = classifier.removeCategory('nonexistent')
    expect(result).toBe(classifier)
  })

  it('classifier still categorizes correctly after removing a category', () => {
    const classifier = bayes()

    classifier.learn('amazing great', 'positive')
    classifier.learn('terrible bad', 'negative')
    classifier.learn('meh ok', 'neutral')

    classifier.removeCategory('neutral')

    const result = classifier.categorize('amazing')
    expect(result.predictedCategory).toBe('positive')
    expect(result.likelihoods.length).toBe(2)
  })

  it('updates vocabulary and vocabularySize correctly', () => {
    const classifier = bayes()

    classifier.learn('unique', 'only')
    const sizeBefore = classifier.vocabularySize

    classifier.removeCategory('only')
    expect(classifier.vocabularySize < sizeBefore).toBeTruthy()
  })

  it('does not produce negative vocabulary counts', () => {
    const classifier = bayes()

    classifier.learn('shared word', 'a')
    classifier.learn('shared word', 'b')

    classifier.removeCategory('a')

    // 'shared' and 'word' should still have count >= 0
    Object.keys(classifier.vocabulary).forEach((token) => {
      expect(classifier.vocabulary[token] >= 0).toBeTruthy()
    })
  })
})

describe('bayes .categorize() return structure', () => {
  it('returns an object with predictedCategory and likelihoods', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')

    const result = classifier.categorize('hello')
    expect(result.hasOwnProperty('predictedCategory')).toBeTruthy()
    expect(result.hasOwnProperty('likelihoods')).toBeTruthy()
    expect(Array.isArray(result.likelihoods)).toBeTruthy()
  })

  it('likelihoods contain category, logLikelihood, logProba, proba', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')

    const result = classifier.categorize('hello')
    const likelihood = result.likelihoods[0]

    expect(likelihood.hasOwnProperty('category')).toBeTruthy()
    expect(likelihood.hasOwnProperty('logLikelihood')).toBeTruthy()
    expect(likelihood.hasOwnProperty('logProba')).toBeTruthy()
    expect(likelihood.hasOwnProperty('proba')).toBeTruthy()
  })

  it('likelihoods are sorted by proba descending', () => {
    const classifier = bayes()
    classifier.learn('aa', 'a')
    classifier.learn('bb', 'b')
    classifier.learn('cc', 'c')

    const result = classifier.categorize('aa')
    for (let i = 1; i < result.likelihoods.length; i++) {
      expect(result.likelihoods[i - 1].proba >= result.likelihoods[i].proba).toBeTruthy()
    }
  })

  it('probabilities sum to approximately 1.0', () => {
    const classifier = bayes()
    classifier.learn('happy fun', 'positive')
    classifier.learn('sad bad', 'negative')

    const result = classifier.categorize('happy')
    const sum = result.likelihoods.reduce((acc, l) => { return acc + l.proba }, 0)
    expect(Math.abs(sum - 1.0) < 0.001).toBeTruthy()
  })

  it('returns predictedCategory null for empty classifier', () => {
    const classifier = bayes()
    const result = classifier.categorize('hello')

    expect(result.predictedCategory).toBe(null)
    expect(result.likelihoods).toEqual([])
  })

  it('throws TypeError when text is not a string', () => {
    const classifier = bayes()
    expect(() => { classifier.categorize(123) }).toThrow(TypeError)
    expect(() => { classifier.categorize(null) }).toThrow(TypeError)
  })
})

describe('bayes edge cases', () => {
  it('handles empty string input to categorize()', () => {
    const classifier = bayes()
    classifier.learn('hello world', 'greetings')

    const result = classifier.categorize('')
    expect(result.hasOwnProperty('predictedCategory')).toBeTruthy()
  })

  it('handles text with only punctuation', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')

    const result = classifier.categorize('!@#$%')
    expect(result.hasOwnProperty('predictedCategory')).toBeTruthy()
  })

  it('handles unknown tokens gracefully', () => {
    const classifier = bayes()
    classifier.learn('hello world', 'greetings')

    const result = classifier.categorize('xyzzy foobar')
    expect(result.hasOwnProperty('predictedCategory')).toBeTruthy()
  })
})

describe('bayes alpha parameter', () => {
  it('uses default alpha of 1 when not specified', () => {
    const classifier = bayes()
    expect(classifier.alpha).toBe(1)
  })

  it('accepts alpha: 0 without overriding to default', () => {
    const classifier = bayes({ alpha: 0 })
    expect(classifier.alpha).toBe(0)
  })

  it('custom alpha affects token probability calculation', () => {
    const classifier1 = bayes({ alpha: 1 })
    const classifier2 = bayes({ alpha: 10 })

    classifier1.learn('hello world', 'greetings')
    classifier1.learn('goodbye world', 'farewells')
    classifier2.learn('hello world', 'greetings')
    classifier2.learn('goodbye world', 'farewells')

    const prob1 = classifier1.tokenProbability('hello', 'greetings')
    const prob2 = classifier2.tokenProbability('hello', 'greetings')

    expect(prob1).not.toBe(prob2)
  })

  it('alpha: 0 categorization works but unseen tokens zero-out a category', () => {
    const classifier = bayes({ alpha: 0 })

    classifier.learn('happy fun', 'positive')
    classifier.learn('sad bad', 'negative')

    // 'happy' was only seen in positive, so negative gets zero probability for it
    const result = classifier.categorize('happy')
    expect(result.predictedCategory).toBe('positive')

    // result should not contain NaN
    result.likelihoods.forEach((l) => {
      expect(!isNaN(l.proba)).toBeTruthy()
    })
  })
})

describe('bayes method chaining', () => {
  it('learn() returns this', () => {
    const classifier = bayes()
    expect(classifier.learn('hello', 'greetings')).toBe(classifier)
  })

  it('initializeCategory() returns this', () => {
    const classifier = bayes()
    expect(classifier.initializeCategory('test')).toBe(classifier)
  })

  it('removeCategory() returns this', () => {
    const classifier = bayes()
    expect(classifier.removeCategory('test')).toBe(classifier)
  })

  it('supports fluent chain: learn().learn().categorize()', () => {
    const result = bayes()
      .learn('happy fun', 'positive')
      .learn('sad bad', 'negative')
      .categorize('happy')

    expect(result.predictedCategory).toBe('positive')
  })
})

describe('bayes .getCategories()', () => {
  it('returns empty array for new classifier', () => {
    const classifier = bayes()
    expect(classifier.getCategories()).toEqual([])
  })

  it('returns array of learned category names', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')
    classifier.learn('bye', 'farewells')

    const categories = classifier.getCategories()
    expect(categories.indexOf('greetings') !== -1).toBeTruthy()
    expect(categories.indexOf('farewells') !== -1).toBeTruthy()
    expect(categories.length).toBe(2)
  })

  it('reflects category removal', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')
    classifier.learn('bye', 'farewells')
    classifier.removeCategory('greetings')

    const categories = classifier.getCategories()
    expect(categories.indexOf('greetings') === -1).toBeTruthy()
    expect(categories.length).toBe(1)
  })
})

describe('bayes .categorizeTopN()', () => {
  it('returns only top N categories', () => {
    const classifier = bayes()
    classifier.learn('aa', 'a')
    classifier.learn('bb', 'b')
    classifier.learn('cc', 'c')

    const result = classifier.categorizeTopN('aa', 2)
    expect(result.likelihoods.length).toBe(2)
  })

  it('returns all categories if N >= total categories', () => {
    const classifier = bayes()
    classifier.learn('aa', 'a')
    classifier.learn('bb', 'b')

    const result = classifier.categorizeTopN('aa', 10)
    expect(result.likelihoods.length).toBe(2)
  })

  it('predictedCategory is the most likely', () => {
    const classifier = bayes()
    classifier.learn('happy fun great', 'positive')
    classifier.learn('sad bad terrible', 'negative')
    classifier.learn('ok meh whatever', 'neutral')

    const result = classifier.categorizeTopN('happy fun', 1)
    expect(result.predictedCategory).toBe('positive')
    expect(result.likelihoods.length).toBe(1)
  })
})

describe('bayes .categorizeWithConfidence()', () => {
  it('returns predictedCategory when above threshold', () => {
    const classifier = bayes()
    classifier.learn('happy fun great amazing', 'positive')
    classifier.learn('sad bad terrible awful', 'negative')

    const result = classifier.categorizeWithConfidence('happy fun great', 0.5)
    expect(result.predictedCategory).toBe('positive')
  })

  it('returns null predictedCategory when below threshold', () => {
    const classifier = bayes()
    classifier.learn('aa', 'a')
    classifier.learn('bb', 'b')

    // with a very high threshold, prediction should be null
    const result = classifier.categorizeWithConfidence('cc', 0.99)
    expect(result.predictedCategory).toBe(null)
  })

  it('returns null for empty classifier', () => {
    const classifier = bayes()
    const result = classifier.categorizeWithConfidence('hello', 0.5)
    expect(result.predictedCategory).toBe(null)
  })

  it('still returns full likelihoods array', () => {
    const classifier = bayes()
    classifier.learn('aa', 'a')
    classifier.learn('bb', 'b')

    const result = classifier.categorizeWithConfidence('cc', 0.99)
    expect(Array.isArray(result.likelihoods)).toBeTruthy()
    expect(result.likelihoods.length > 0).toBeTruthy()
  })

  it('throws TypeError for invalid threshold', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')

    expect(() => { classifier.categorizeWithConfidence('hello', -1) }).toThrow(TypeError)
    expect(() => { classifier.categorizeWithConfidence('hello', 2) }).toThrow(TypeError)
    expect(() => { classifier.categorizeWithConfidence('hello', 'bad') }).toThrow(TypeError)
  })
})

describe('bayes .topInfluentialTokens()', () => {
  it('returns top tokens for a classification', () => {
    const classifier = bayes()
    classifier.learn('happy fun great joy', 'positive')
    classifier.learn('sad bad terrible gloom', 'negative')

    const tokens = classifier.topInfluentialTokens('happy fun great', 3)
    expect(Array.isArray(tokens)).toBeTruthy()
    expect(tokens.length <= 3).toBeTruthy()
    expect(tokens.length > 0).toBeTruthy()

    // each token should have the right shape
    tokens.forEach((t) => {
      expect(t.hasOwnProperty('token')).toBeTruthy()
      expect(t.hasOwnProperty('probability')).toBeTruthy()
      expect(t.hasOwnProperty('frequency')).toBeTruthy()
    })
  })

  it('returns empty array for empty classifier', () => {
    const classifier = bayes()
    const tokens = classifier.topInfluentialTokens('hello')
    expect(tokens).toEqual([])
  })

  it('tokens are sorted by probability descending', () => {
    const classifier = bayes()
    classifier.learn('apple banana cherry', 'fruit')
    classifier.learn('dog cat bird', 'animal')

    const tokens = classifier.topInfluentialTokens('apple banana cherry', 5)
    for (let i = 1; i < tokens.length; i++) {
      expect(tokens[i - 1].probability >= tokens[i].probability).toBeTruthy()
    }
  })

  it('defaults to 5 tokens', () => {
    const classifier = bayes()
    classifier.learn('a b c d e f g h', 'letters')
    classifier.learn('1 2 3', 'numbers')

    const tokens = classifier.topInfluentialTokens('a b c d e f g h')
    expect(tokens.length <= 5).toBeTruthy()
  })

  it('returns empty array when n is 0', () => {
    const classifier = bayes()
    classifier.learn('hello world', 'greetings')

    const tokens = classifier.topInfluentialTokens('hello world', 0)
    expect(tokens).toEqual([])
  })

  it('handles negative n by returning empty array', () => {
    const classifier = bayes()
    classifier.learn('hello world', 'greetings')

    const tokens = classifier.topInfluentialTokens('hello world', -3)
    expect(tokens).toEqual([])
  })
})

describe('bayes .learnBatch()', () => {
  it('learns multiple items at once', () => {
    const classifier = bayes()
    classifier.learnBatch([
      { text: 'happy fun', category: 'positive' },
      { text: 'sad bad', category: 'negative' }
    ])

    expect(classifier.totalDocuments).toBe(2)
    expect(classifier.categories['positive']).toBeTruthy()
    expect(classifier.categories['negative']).toBeTruthy()
  })

  it('produces same result as individual learn calls', () => {
    const classifier1 = bayes()
    classifier1.learn('happy fun', 'positive')
    classifier1.learn('sad bad', 'negative')

    const classifier2 = bayes()
    classifier2.learnBatch([
      { text: 'happy fun', category: 'positive' },
      { text: 'sad bad', category: 'negative' }
    ])

    expect(classifier1.vocabulary).toEqual(classifier2.vocabulary)
    expect(classifier1.totalDocuments).toBe(classifier2.totalDocuments)
    expect(classifier1.docCount).toEqual(classifier2.docCount)
  })

  it('throws TypeError on non-array input', () => {
    const classifier = bayes()
    expect(() => { classifier.learnBatch('not an array') }).toThrow(TypeError)
    expect(() => { classifier.learnBatch(42) }).toThrow(TypeError)
  })

  it('returns this for chaining', () => {
    const classifier = bayes()
    const result = classifier.learnBatch([{ text: 'hello', category: 'greetings' }])
    expect(result).toBe(classifier)
  })
})

describe('bayes .reset()', () => {
  it('clears all learned data', () => {
    const classifier = bayes()
    classifier.learn('hello world', 'greetings')
    classifier.learn('goodbye world', 'farewells')

    classifier.reset()

    expect(classifier.totalDocuments).toBe(0)
    expect(classifier.vocabularySize).toBe(0)
    expect(classifier.categories).toEqual({})
    expect(classifier.vocabulary).toEqual({})
    expect(classifier.docCount).toEqual({})
    expect(classifier.wordCount).toEqual({})
    expect(classifier.wordFrequencyCount).toEqual({})
  })

  it('preserves options (tokenizer, alpha, fitPrior)', () => {
    const customTokenizer = (text) => { return text.split('') }
    const classifier = bayes({ tokenizer: customTokenizer, alpha: 2, fitPrior: false })
    classifier.learn('abc', 'letters')

    classifier.reset()

    expect(classifier.tokenizer).toBe(customTokenizer)
    expect(classifier.alpha).toBe(2)
    expect(classifier.fitPrior).toBe(false)
  })

  it('classifier can be retrained after reset', () => {
    const classifier = bayes()
    classifier.learn('hello', 'greetings')
    classifier.reset()
    classifier.learn('goodbye', 'farewells')

    expect(classifier.totalDocuments).toBe(1)
    expect(classifier.getCategories()).toEqual(['farewells'])
  })

  it('returns this for chaining', () => {
    const classifier = bayes()
    expect(classifier.reset()).toBe(classifier)
  })
})

describe('bayes .getCategoryStats()', () => {
  it('returns correct doc and word counts per category', () => {
    const classifier = bayes()
    classifier.learn('hello world', 'greetings')
    classifier.learn('goodbye world', 'farewells')
    classifier.learn('hi there', 'greetings')

    const stats = classifier.getCategoryStats()

    expect(stats.greetings.docCount).toBe(2)
    expect(stats.farewells.docCount).toBe(1)
    expect(stats.greetings.wordCount > 0).toBeTruthy()
    expect(stats.greetings.vocabularySize > 0).toBeTruthy()
  })

  it('includes _total aggregate stats with wordCount', () => {
    const classifier = bayes()
    classifier.learn('hello world', 'greetings')
    classifier.learn('bye now', 'farewells')

    const stats = classifier.getCategoryStats()

    expect(stats._total.docCount).toBe(2)
    expect(stats._total.vocabularySize > 0).toBeTruthy()
    expect(stats._total.wordCount).toBe(stats.greetings.wordCount + stats.farewells.wordCount)
  })
})

describe('bayes numerical stability (logsumexp)', () => {
  it('probabilities still sum to ~1.0 with many categories', () => {
    const classifier = bayes()
    for (let i = 0; i < 20; i++) {
      classifier.learn('word' + i + ' common shared text', 'cat' + i)
    }

    const result = classifier.categorize('word0 common shared')
    const sum = result.likelihoods.reduce((acc, l) => { return acc + l.proba }, 0)
    expect(Math.abs(sum - 1.0) < 0.01).toBeTruthy()
  })

  it('handles long documents without NaN', () => {
    const classifier = bayes()
    classifier.learn('good great amazing wonderful fantastic', 'positive')
    classifier.learn('bad terrible awful horrible dreadful', 'negative')

    // create a long document
    let longText = ''
    for (let i = 0; i < 100; i++) {
      longText += 'good great amazing '
    }

    const result = classifier.categorize(longText)
    expect(!isNaN(result.likelihoods[0].proba)).toBeTruthy()
    expect(result.likelihoods[0].proba > 0).toBeTruthy()
    expect(result.predictedCategory).toBe('positive')
  })
})
