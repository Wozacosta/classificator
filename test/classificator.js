var assert = require('assert')
  , bayes = require('../lib/classificator')

describe('bayes() init', function () {
  it('valid options (falsey or with an object) do not raise Errors', function () {
    var validOptionsCases = [ undefined, {} ];

    validOptionsCases.forEach(function (validOptions) {
      var classifier = bayes(validOptions)
      assert.deepEqual(classifier.options, {})
    })
  })

  it('invalid options (truthy and not object) raise TypeError during init', function () {
    var invalidOptionsCases = [ null, 0, 'a', [] ];

    invalidOptionsCases.forEach(function (invalidOptions) {
      assert.throws(function () { bayes(invalidOptions) }, Error)
      assert.throws(function () { bayes(invalidOptions) }, TypeError)
    })
  })
})

describe('bayes using custom tokenizer', function () {
  it('uses custom tokenization function if one is provided in `options`.', function () {
    var splitOnChar = function (text) {
      return text.split('')
    }

    var classifier = bayes({ tokenizer: splitOnChar })

    classifier.learn('abcd', 'happy')

    assert.equal(classifier.totalDocuments, 1)
    assert.equal(classifier.docCount.happy, 1)
    assert.deepEqual(classifier.vocabulary, { a: 1, b: 1, c: 1, d: 1 })
    assert.equal(classifier.vocabularySize, 4)
    assert.equal(classifier.wordCount.happy, 4)
    assert.equal(classifier.wordFrequencyCount.happy.a, 1)
    assert.equal(classifier.wordFrequencyCount.happy.b, 1)
    assert.equal(classifier.wordFrequencyCount.happy.c, 1)
    assert.equal(classifier.wordFrequencyCount.happy.d, 1)
    assert.deepStrictEqual(classifier.categories, { happy: true })
  })
})

describe('bayes serializing/deserializing its state', function () {
  it('serializes/deserializes its state as JSON correctly.', function () {
    var classifier = bayes()

    classifier.learn('Fun times were had by all', 'positive')
    classifier.learn('sad dark rainy day in the cave', 'negative')

    var jsonRepr = classifier.toJson()
    var state = JSON.parse(jsonRepr)

    bayes.STATE_KEYS.forEach(function (k) {
      assert.deepEqual(state[k], classifier[k])
    })

    var revivedClassifier = bayes.fromJson(jsonRepr)

    bayes.STATE_KEYS.forEach(function (k) {
      assert.deepEqual(revivedClassifier[k], classifier[k])
    })
  })
})

describe('bayes using custom tokenizer with fromJson', function () {
  it('accepts a custom tokenizer passed as an option to fromJson', function () {
    var splitOnChar = function (text) {
      return text.split('')
    }

    var classifier = bayes({ tokenizer: splitOnChar })

    classifier.learn('abcd', 'happy')
    classifier.learn('efgh', 'sad')

    var jsonRepr = classifier.toJson()
    var revivedClassifier = bayes.fromJson(jsonRepr, { tokenizer: splitOnChar })

    var result = revivedClassifier.categorize('abcd')
    assert.equal(result.predictedCategory, 'happy')
  })
})

describe('bayes .fromJson() edge cases', function () {
  it('throws on null input', function () {
    assert.throws(function () { bayes.fromJson(null) }, Error)
  })

  it('throws on numeric input', function () {
    assert.throws(function () { bayes.fromJson(42) }, Error)
  })

  it('throws on invalid JSON string', function () {
    assert.throws(function () { bayes.fromJson('not valid json') }, Error)
  })

  it('throws when JSON is missing required state keys', function () {
    assert.throws(function () { bayes.fromJson('{"options":{}}') }, Error)
  })

  it('preserves options through serialization round-trip', function () {
    var classifier = bayes({ alpha: 2, fitPrior: false })
    classifier.learn('hello world', 'greetings')

    var revived = bayes.fromJson(classifier.toJson())
    assert.equal(revived.alpha, 2)
    assert.equal(revived.fitPrior, false)
  })
})

describe('bayes .learn() correctness', function () {
  it('categorizes correctly for `positive` and `negative` categories', function () {
    let classifier = bayes();

    classifier.learn('amazing, awesome movie!! Yeah!!', 'positive')
    classifier.learn('Sweet, this is incredibly, amazing, perfect, great!!', 'positive')
    classifier.learn('terrible, shitty thing. Damn. Sucks!!', 'negative')
    classifier.learn('I dont really know what to make of this.', 'neutral')

    assert.deepEqual(classifier.categorize('awesome, cool, amazing!! Yay.').predictedCategory, 'positive')
  })

  it('categorizes correctly for `chinese` and `japanese` categories', function () {
    var classifier = bayes()

    classifier.learn('Chinese Beijing Chinese', 'chinese')
    classifier.learn('Chinese Chinese Shanghai', 'chinese')
    classifier.learn('Chinese Macao', 'chinese')
    classifier.learn('Tokyo Japan Chinese', 'japanese')

    var chineseFrequencyCount = classifier.wordFrequencyCount.chinese

    assert.equal(chineseFrequencyCount['Chinese'], 5)
    assert.equal(chineseFrequencyCount['Beijing'], 1)
    assert.equal(chineseFrequencyCount['Shanghai'], 1)
    assert.equal(chineseFrequencyCount['Macao'], 1)

    var japaneseFrequencyCount = classifier.wordFrequencyCount.japanese

    assert.equal(japaneseFrequencyCount['Tokyo'], 1)
    assert.equal(japaneseFrequencyCount['Japan'], 1)
    assert.equal(japaneseFrequencyCount['Chinese'], 1)

    assert.deepEqual(classifier.categorize('Chinese Chinese Chinese Tokyo Japan').predictedCategory,'chinese')
  })

  it('correctly tokenizes cyrlic characters', function () {
    var classifier = bayes()

    classifier.learn('Надежда за', 'a')
    classifier.learn('Надежда за обич еп.36 Тест', 'b')
    classifier.learn('Надежда за обич еп.36 Тест', 'b')

    var aFreqCount = classifier.wordFrequencyCount.a
    assert.equal(aFreqCount['Надежда'], 1)
    assert.equal(aFreqCount['за'], 1)

    var bFreqCount = classifier.wordFrequencyCount.b
    assert.equal(bFreqCount['Надежда'], 2)
    assert.equal(bFreqCount['за'], 2)
    assert.equal(bFreqCount['обич'], 2)
    assert.equal(bFreqCount['еп'], 2)
    assert.equal(bFreqCount['36'], 2)
    assert.equal(bFreqCount['Тест'], 2)
  })

  it('correctly computes probabilities without prior', function () {
    var classifier = bayes({ fitPrior: false})

    classifier.learn('aa', '1')
    classifier.learn('aa', '1')
    classifier.learn('aa', '1')
    classifier.learn('bb', '2')

    assert.equal(classifier.categorize('cc').likelihoods[0].proba, 0.5)
    assert.equal(Number(classifier.categorize('bb').likelihoods[0].proba).toFixed(6), Number(0.76923077).toFixed(6))
    assert.equal(Number(classifier.categorize('aa').likelihoods[0].proba).toFixed(6), Number(0.70588235).toFixed(6))
  })

  it('correctly computes probabilities with prior', function () {
    var classifier = bayes()

    classifier.learn('aa', '1')
    classifier.learn('aa', '1')
    classifier.learn('aa', '1')
    classifier.learn('bb', '2')

    assert.equal(classifier.categorize('cc').likelihoods[0].proba, 0.75)
    assert.equal(Number(classifier.categorize('bb').likelihoods[0].proba).toFixed(6), Number(0.52631579).toFixed(6))
    assert.equal(Number(classifier.categorize('aa').likelihoods[0].proba).toFixed(6), Number(0.87804878).toFixed(6))
  })

  it('throws TypeError when text is not a string', function () {
    var classifier = bayes()
    assert.throws(function () { classifier.learn(123, 'cat') }, TypeError)
    assert.throws(function () { classifier.learn(null, 'cat') }, TypeError)
  })

  it('throws TypeError when category is not a string', function () {
    var classifier = bayes()
    assert.throws(function () { classifier.learn('hello', 123) }, TypeError)
    assert.throws(function () { classifier.learn('hello', null) }, TypeError)
  })
})

describe('bayes .unlearn() correctness', function () {
  it('reverses the effect of a single learn call', function () {
    var classifier = bayes()

    classifier.learn('fun times', 'positive')
    classifier.learn('bad times', 'negative')
    classifier.learn('great day', 'positive')

    var docsBefore = classifier.totalDocuments
    classifier.unlearn('fun times', 'positive')

    assert.equal(classifier.totalDocuments, docsBefore - 1)
  })

  it('throws when unlearning from a non-existent category', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')

    assert.throws(function () {
      classifier.unlearn('hello', 'nonexistent')
    }, Error)
  })

  it('removes category from categories when last doc is unlearned', function () {
    var classifier = bayes()

    classifier.learn('hello', 'greetings')
    assert.ok(classifier.categories['greetings'])

    classifier.unlearn('hello', 'greetings')
    assert.equal(classifier.categories['greetings'], undefined)
  })

  it('classifier still works correctly after unlearn', function () {
    var classifier = bayes()

    classifier.learn('amazing great', 'positive')
    classifier.learn('terrible awful', 'negative')
    classifier.learn('bad horrible', 'negative')

    classifier.unlearn('bad horrible', 'negative')

    var result = classifier.categorize('terrible')
    assert.equal(result.predictedCategory, 'negative')
  })

  it('returns this for method chaining', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')
    var result = classifier.unlearn('hello', 'greetings')
    assert.strictEqual(result, classifier)
  })

  it('throws TypeError when text is not a string', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')
    assert.throws(function () { classifier.unlearn(123, 'greetings') }, TypeError)
  })
})

describe('bayes .removeCategory()', function () {
  it('removes a category and its associated data', function () {
    var classifier = bayes()

    classifier.learn('hello world', 'greetings')
    classifier.learn('bad stuff', 'negative')

    classifier.removeCategory('greetings')

    assert.equal(classifier.categories['greetings'], undefined)
    assert.equal(classifier.docCount['greetings'], undefined)
    assert.equal(classifier.wordCount['greetings'], undefined)
    assert.equal(classifier.wordFrequencyCount['greetings'], undefined)
  })

  it('returns this for chaining', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')
    var result = classifier.removeCategory('greetings')
    assert.strictEqual(result, classifier)
  })

  it('returns this when removing non-existent category (no-op)', function () {
    var classifier = bayes()
    var result = classifier.removeCategory('nonexistent')
    assert.strictEqual(result, classifier)
  })

  it('classifier still categorizes correctly after removing a category', function () {
    var classifier = bayes()

    classifier.learn('amazing great', 'positive')
    classifier.learn('terrible bad', 'negative')
    classifier.learn('meh ok', 'neutral')

    classifier.removeCategory('neutral')

    var result = classifier.categorize('amazing')
    assert.equal(result.predictedCategory, 'positive')
    assert.equal(result.likelihoods.length, 2)
  })

  it('updates vocabulary and vocabularySize correctly', function () {
    var classifier = bayes()

    classifier.learn('unique', 'only')
    var sizeBefore = classifier.vocabularySize

    classifier.removeCategory('only')
    assert.ok(classifier.vocabularySize < sizeBefore)
  })
})

describe('bayes .categorize() return structure', function () {
  it('returns an object with predictedCategory and likelihoods', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')

    var result = classifier.categorize('hello')
    assert.ok(result.hasOwnProperty('predictedCategory'))
    assert.ok(result.hasOwnProperty('likelihoods'))
    assert.ok(Array.isArray(result.likelihoods))
  })

  it('likelihoods contain category, logLikelihood, logProba, proba', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')

    var result = classifier.categorize('hello')
    var likelihood = result.likelihoods[0]

    assert.ok(likelihood.hasOwnProperty('category'))
    assert.ok(likelihood.hasOwnProperty('logLikelihood'))
    assert.ok(likelihood.hasOwnProperty('logProba'))
    assert.ok(likelihood.hasOwnProperty('proba'))
  })

  it('likelihoods are sorted by proba descending', function () {
    var classifier = bayes()
    classifier.learn('aa', 'a')
    classifier.learn('bb', 'b')
    classifier.learn('cc', 'c')

    var result = classifier.categorize('aa')
    for (var i = 1; i < result.likelihoods.length; i++) {
      assert.ok(result.likelihoods[i - 1].proba >= result.likelihoods[i].proba)
    }
  })

  it('probabilities sum to approximately 1.0', function () {
    var classifier = bayes()
    classifier.learn('happy fun', 'positive')
    classifier.learn('sad bad', 'negative')

    var result = classifier.categorize('happy')
    var sum = result.likelihoods.reduce(function (acc, l) { return acc + l.proba }, 0)
    assert.ok(Math.abs(sum - 1.0) < 0.001)
  })

  it('returns predictedCategory null for empty classifier', function () {
    var classifier = bayes()
    var result = classifier.categorize('hello')

    assert.equal(result.predictedCategory, null)
    assert.deepEqual(result.likelihoods, [])
  })

  it('throws TypeError when text is not a string', function () {
    var classifier = bayes()
    assert.throws(function () { classifier.categorize(123) }, TypeError)
    assert.throws(function () { classifier.categorize(null) }, TypeError)
  })
})

describe('bayes edge cases', function () {
  it('handles empty string input to categorize()', function () {
    var classifier = bayes()
    classifier.learn('hello world', 'greetings')

    var result = classifier.categorize('')
    assert.ok(result.hasOwnProperty('predictedCategory'))
  })

  it('handles text with only punctuation', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')

    var result = classifier.categorize('!@#$%')
    assert.ok(result.hasOwnProperty('predictedCategory'))
  })

  it('handles unknown tokens gracefully', function () {
    var classifier = bayes()
    classifier.learn('hello world', 'greetings')

    var result = classifier.categorize('xyzzy foobar')
    assert.ok(result.hasOwnProperty('predictedCategory'))
  })
})

describe('bayes alpha parameter', function () {
  it('uses default alpha of 1 when not specified', function () {
    var classifier = bayes()
    assert.equal(classifier.alpha, 1)
  })

  it('accepts alpha: 0 without overriding to default', function () {
    var classifier = bayes({ alpha: 0 })
    assert.strictEqual(classifier.alpha, 0)
  })

  it('custom alpha affects token probability calculation', function () {
    var classifier1 = bayes({ alpha: 1 })
    var classifier2 = bayes({ alpha: 10 })

    classifier1.learn('hello world', 'greetings')
    classifier1.learn('goodbye world', 'farewells')
    classifier2.learn('hello world', 'greetings')
    classifier2.learn('goodbye world', 'farewells')

    var prob1 = classifier1.tokenProbability('hello', 'greetings')
    var prob2 = classifier2.tokenProbability('hello', 'greetings')

    assert.notEqual(prob1, prob2)
  })
})

describe('bayes method chaining', function () {
  it('learn() returns this', function () {
    var classifier = bayes()
    assert.strictEqual(classifier.learn('hello', 'greetings'), classifier)
  })

  it('initializeCategory() returns this', function () {
    var classifier = bayes()
    assert.strictEqual(classifier.initializeCategory('test'), classifier)
  })

  it('removeCategory() returns this', function () {
    var classifier = bayes()
    assert.strictEqual(classifier.removeCategory('test'), classifier)
  })

  it('supports fluent chain: learn().learn().categorize()', function () {
    var result = bayes()
      .learn('happy fun', 'positive')
      .learn('sad bad', 'negative')
      .categorize('happy')

    assert.equal(result.predictedCategory, 'positive')
  })
})

describe('bayes .getCategories()', function () {
  it('returns empty array for new classifier', function () {
    var classifier = bayes()
    assert.deepEqual(classifier.getCategories(), [])
  })

  it('returns array of learned category names', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')
    classifier.learn('bye', 'farewells')

    var categories = classifier.getCategories()
    assert.ok(categories.indexOf('greetings') !== -1)
    assert.ok(categories.indexOf('farewells') !== -1)
    assert.equal(categories.length, 2)
  })

  it('reflects category removal', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')
    classifier.learn('bye', 'farewells')
    classifier.removeCategory('greetings')

    var categories = classifier.getCategories()
    assert.ok(categories.indexOf('greetings') === -1)
    assert.equal(categories.length, 1)
  })
})

describe('bayes .categorizeTopN()', function () {
  it('returns only top N categories', function () {
    var classifier = bayes()
    classifier.learn('aa', 'a')
    classifier.learn('bb', 'b')
    classifier.learn('cc', 'c')

    var result = classifier.categorizeTopN('aa', 2)
    assert.equal(result.likelihoods.length, 2)
  })

  it('returns all categories if N >= total categories', function () {
    var classifier = bayes()
    classifier.learn('aa', 'a')
    classifier.learn('bb', 'b')

    var result = classifier.categorizeTopN('aa', 10)
    assert.equal(result.likelihoods.length, 2)
  })

  it('predictedCategory is the most likely', function () {
    var classifier = bayes()
    classifier.learn('happy fun great', 'positive')
    classifier.learn('sad bad terrible', 'negative')
    classifier.learn('ok meh whatever', 'neutral')

    var result = classifier.categorizeTopN('happy fun', 1)
    assert.equal(result.predictedCategory, 'positive')
    assert.equal(result.likelihoods.length, 1)
  })
})

describe('bayes .learnBatch()', function () {
  it('learns multiple items at once', function () {
    var classifier = bayes()
    classifier.learnBatch([
      { text: 'happy fun', category: 'positive' },
      { text: 'sad bad', category: 'negative' }
    ])

    assert.equal(classifier.totalDocuments, 2)
    assert.ok(classifier.categories['positive'])
    assert.ok(classifier.categories['negative'])
  })

  it('produces same result as individual learn calls', function () {
    var classifier1 = bayes()
    classifier1.learn('happy fun', 'positive')
    classifier1.learn('sad bad', 'negative')

    var classifier2 = bayes()
    classifier2.learnBatch([
      { text: 'happy fun', category: 'positive' },
      { text: 'sad bad', category: 'negative' }
    ])

    assert.deepEqual(classifier1.vocabulary, classifier2.vocabulary)
    assert.equal(classifier1.totalDocuments, classifier2.totalDocuments)
    assert.deepEqual(classifier1.docCount, classifier2.docCount)
  })

  it('throws TypeError on non-array input', function () {
    var classifier = bayes()
    assert.throws(function () { classifier.learnBatch('not an array') }, TypeError)
    assert.throws(function () { classifier.learnBatch(42) }, TypeError)
  })

  it('returns this for chaining', function () {
    var classifier = bayes()
    var result = classifier.learnBatch([{ text: 'hello', category: 'greetings' }])
    assert.strictEqual(result, classifier)
  })
})

describe('bayes .reset()', function () {
  it('clears all learned data', function () {
    var classifier = bayes()
    classifier.learn('hello world', 'greetings')
    classifier.learn('goodbye world', 'farewells')

    classifier.reset()

    assert.equal(classifier.totalDocuments, 0)
    assert.equal(classifier.vocabularySize, 0)
    assert.deepEqual(classifier.categories, {})
    assert.deepEqual(classifier.vocabulary, {})
    assert.deepEqual(classifier.docCount, {})
    assert.deepEqual(classifier.wordCount, {})
    assert.deepEqual(classifier.wordFrequencyCount, {})
  })

  it('preserves options (tokenizer, alpha, fitPrior)', function () {
    var customTokenizer = function (text) { return text.split('') }
    var classifier = bayes({ tokenizer: customTokenizer, alpha: 2, fitPrior: false })
    classifier.learn('abc', 'letters')

    classifier.reset()

    assert.strictEqual(classifier.tokenizer, customTokenizer)
    assert.equal(classifier.alpha, 2)
    assert.equal(classifier.fitPrior, false)
  })

  it('classifier can be retrained after reset', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')
    classifier.reset()
    classifier.learn('goodbye', 'farewells')

    assert.equal(classifier.totalDocuments, 1)
    assert.deepEqual(classifier.getCategories(), ['farewells'])
  })

  it('returns this for chaining', function () {
    var classifier = bayes()
    assert.strictEqual(classifier.reset(), classifier)
  })
})

describe('bayes .getCategoryStats()', function () {
  it('returns correct doc and word counts per category', function () {
    var classifier = bayes()
    classifier.learn('hello world', 'greetings')
    classifier.learn('goodbye world', 'farewells')
    classifier.learn('hi there', 'greetings')

    var stats = classifier.getCategoryStats()

    assert.equal(stats.greetings.docCount, 2)
    assert.equal(stats.farewells.docCount, 1)
    assert.ok(stats.greetings.wordCount > 0)
    assert.ok(stats.greetings.vocabularySize > 0)
  })

  it('includes _total aggregate stats', function () {
    var classifier = bayes()
    classifier.learn('hello', 'greetings')
    classifier.learn('bye', 'farewells')

    var stats = classifier.getCategoryStats()

    assert.equal(stats._total.docCount, 2)
    assert.ok(stats._total.vocabularySize > 0)
  })
})
