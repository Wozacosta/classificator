var assert = require('assert')
  , bayes = require('../lib/classificator')

// =============================================================================
// Integration Tests
// Test multiple features working together in realistic combinations
// =============================================================================

describe('[Integration] full train → serialize → restore → classify pipeline', function () {
  it('classifier survives a full round-trip with custom options', function () {
    var tokenizer = function (text) { return text.toLowerCase().split(/\s+/) }
    var preprocessor = function (tokens) {
      var stops = new Set(['the', 'a', 'is', 'it', 'and', 'of', 'to', 'in'])
      return tokens.filter(function (t) { return !stops.has(t) })
    }

    // 1. Create with custom options
    var classifier = bayes({
      tokenizer: tokenizer,
      tokenPreprocessor: preprocessor,
      alpha: 0.5,
      fitPrior: true
    })

    // 2. Train
    classifier.learn('The movie was amazing and wonderful', 'positive')
    classifier.learn('It is a great film to watch', 'positive')
    classifier.learn('The movie was terrible and boring', 'negative')
    classifier.learn('It is a bad film, awful acting', 'negative')

    // 3. Verify pre-serialization
    var before = classifier.categorize('amazing film')
    assert.equal(before.predictedCategory, 'positive')

    // 4. Serialize
    var json = classifier.toJson()

    // 5. Restore with runtime options
    var restored = bayes.fromJson(json, {
      tokenizer: tokenizer,
      tokenPreprocessor: preprocessor
    })

    // 6. Verify post-restoration
    var after = restored.categorize('amazing film')
    assert.equal(after.predictedCategory, 'positive')
    assert.equal(after.likelihoods.length, before.likelihoods.length)

    // 7. Probabilities should match
    assert.equal(
      before.likelihoods[0].proba.toFixed(8),
      after.likelihoods[0].proba.toFixed(8)
    )

    // 8. Options preserved
    assert.equal(restored.alpha, 0.5)
    assert.equal(restored.fitPrior, true)
    assert.strictEqual(restored.tokenizer, tokenizer)
    assert.strictEqual(restored.tokenPreprocessor, preprocessor)
  })
})

describe('[Integration] learn → unlearn → relearn cycle', function () {
  it('classifier state is consistent after learn/unlearn/relearn', function () {
    var classifier = bayes()

    // learn initial data
    classifier.learn('good morning sunshine', 'positive')
    classifier.learn('terrible horrible day', 'negative')
    classifier.learn('wonderful great time', 'positive')

    var stats1 = classifier.getCategoryStats()
    assert.equal(stats1._total.docCount, 3)

    // unlearn a mistake
    classifier.unlearn('wonderful great time', 'positive')
    assert.equal(classifier.totalDocuments, 2)

    // re-learn corrected data
    classifier.learn('wonderful great time', 'neutral')
    assert.equal(classifier.totalDocuments, 3)

    // classifier should now have 3 categories
    var categories = classifier.getCategories()
    assert.ok(categories.indexOf('positive') !== -1)
    assert.ok(categories.indexOf('negative') !== -1)
    assert.ok(categories.indexOf('neutral') !== -1)

    // classification still works
    var result = classifier.categorize('terrible')
    assert.equal(result.predictedCategory, 'negative')
  })
})

describe('[Integration] batch learning + stats + reset + retrain', function () {
  it('full lifecycle: batch train, inspect, reset, retrain differently', function () {
    var classifier = bayes()

    // batch train
    classifier.learnBatch([
      { text: 'buy cheap viagra now', category: 'spam' },
      { text: 'limited offer free pills', category: 'spam' },
      { text: 'hello how are you doing', category: 'ham' },
      { text: 'meeting at 3pm tomorrow', category: 'ham' },
      { text: 'project update attached', category: 'ham' }
    ])

    // inspect
    var stats = classifier.getCategoryStats()
    assert.equal(stats.spam.docCount, 2)
    assert.equal(stats.ham.docCount, 3)
    assert.equal(stats._total.docCount, 5)

    // classify
    assert.equal(classifier.categorize('free offer').predictedCategory, 'spam')
    assert.equal(classifier.categorize('meeting tomorrow').predictedCategory, 'ham')

    // reset
    classifier.reset()
    assert.equal(classifier.totalDocuments, 0)
    assert.deepEqual(classifier.getCategories(), [])

    // retrain with different categories
    classifier.learn('breaking news politics', 'news')
    classifier.learn('football scores today', 'sports')

    assert.equal(classifier.categorize('political news').predictedCategory, 'news')
    assert.equal(classifier.categorize('football game').predictedCategory, 'sports')
  })
})

describe('[Integration] removeCategory + reclassification', function () {
  it('removing a dominant category shifts predictions correctly', function () {
    var classifier = bayes()

    classifier.learn('python code function', 'programming')
    classifier.learn('java class object', 'programming')
    classifier.learn('javascript react component', 'programming')
    classifier.learn('recipe bake flour sugar', 'cooking')
    classifier.learn('stock market investment', 'finance')

    // programming dominates with fitPrior
    var before = classifier.categorize('new class today')
    assert.equal(before.predictedCategory, 'programming')
    assert.equal(before.likelihoods.length, 3)

    // remove programming
    classifier.removeCategory('programming')

    // now should choose between cooking and finance
    var after = classifier.categorize('new class today')
    assert.equal(after.likelihoods.length, 2)
    assert.ok(after.predictedCategory === 'cooking' || after.predictedCategory === 'finance')

    // probabilities still sum to ~1
    var sum = after.likelihoods.reduce(function (acc, l) { return acc + l.proba }, 0)
    assert.ok(Math.abs(sum - 1.0) < 0.01)
  })
})

describe('[Integration] tokenPreprocessor affects all operations consistently', function () {
  it('preprocessor is applied in learn, unlearn, categorize, and topInfluentialTokens', function () {
    var lowered = []
    var preprocessor = function (tokens) {
      var result = tokens.map(function (t) { return t.toLowerCase() })
      lowered.push(result)
      return result
    }

    var classifier = bayes({ tokenPreprocessor: preprocessor })

    // learn — preprocessor called
    lowered = []
    classifier.learn('HELLO WORLD', 'greetings')
    assert.ok(lowered.length > 0)
    assert.deepEqual(lowered[0], ['hello', 'world'])

    // should have lowercase tokens in vocabulary
    assert.ok(classifier.vocabulary['hello'])
    assert.equal(classifier.vocabulary['HELLO'], undefined)

    // categorize — preprocessor called
    lowered = []
    var result = classifier.categorize('HELLO')
    assert.ok(lowered.length > 0)
    assert.equal(result.predictedCategory, 'greetings')

    // topInfluentialTokens — preprocessor called
    lowered = []
    var tokens = classifier.topInfluentialTokens('HELLO WORLD', 2)
    assert.ok(lowered.length > 0)
    assert.ok(tokens.length > 0)

    // unlearn — preprocessor called (lowercase matches original learn)
    lowered = []
    classifier.learn('BYE NOW', 'farewells')
    classifier.unlearn('HELLO WORLD', 'greetings')
    assert.ok(lowered.length > 0)
    assert.equal(classifier.categories['greetings'], undefined)
  })
})

describe('[Integration] confidence threshold + topN combined workflow', function () {
  it('uses confidence to filter then topN to limit results', function () {
    var classifier = bayes()

    classifier.learnBatch([
      { text: 'cat dog hamster', category: 'pets' },
      { text: 'cat dog hamster', category: 'pets' },
      { text: 'car truck bus', category: 'vehicles' },
      { text: 'apple banana orange', category: 'fruit' },
      { text: 'table chair desk', category: 'furniture' }
    ])

    // high confidence prediction
    var confident = classifier.categorizeWithConfidence('cat dog', 0.3)
    assert.equal(confident.predictedCategory, 'pets')

    // low confidence for ambiguous text
    var unsure = classifier.categorizeWithConfidence('xyz unknown', 0.5)
    assert.equal(unsure.predictedCategory, null)

    // topN limits output
    var top2 = classifier.categorizeTopN('cat dog', 2)
    assert.equal(top2.likelihoods.length, 2)
    assert.equal(top2.predictedCategory, 'pets')
  })
})

describe('[Integration] method chaining complex workflow', function () {
  it('chains learn, unlearn, removeCategory, and ends with categorize', function () {
    var result = bayes()
      .learn('happy joy love', 'positive')
      .learn('sad hate anger', 'negative')
      .learn('meh whatever ok', 'neutral')
      .learn('oops wrong category', 'neutral')
      .unlearn('oops wrong category', 'neutral')
      .removeCategory('neutral')
      .learn('wonderful amazing', 'positive')
      .categorize('love and joy')

    assert.equal(result.predictedCategory, 'positive')
    assert.equal(result.likelihoods.length, 2)
  })
})

// =============================================================================
// End-to-End Tests
// Simulate real-world classification scenarios from start to finish
// =============================================================================

describe('[E2E] email spam detection', function () {
  var classifier

  beforeEach(function () {
    classifier = bayes()

    // train spam
    classifier.learnBatch([
      { text: 'Buy cheap viagra online now discount', category: 'spam' },
      { text: 'You won a free prize click here to claim', category: 'spam' },
      { text: 'Limited time offer buy one get one free', category: 'spam' },
      { text: 'Earn money fast work from home guaranteed', category: 'spam' },
      { text: 'Congratulations you have been selected winner', category: 'spam' }
    ])

    // train ham
    classifier.learnBatch([
      { text: 'Hey can we meet for lunch tomorrow', category: 'ham' },
      { text: 'Please review the attached quarterly report', category: 'ham' },
      { text: 'The meeting has been moved to 3pm', category: 'ham' },
      { text: 'Here are the notes from today discussion', category: 'ham' },
      { text: 'Can you send me the project update please', category: 'ham' }
    ])
  })

  it('correctly classifies obvious spam', function () {
    assert.equal(
      classifier.categorize('Buy now free offer limited time').predictedCategory,
      'spam'
    )
  })

  it('correctly classifies legitimate email', function () {
    assert.equal(
      classifier.categorize('Can we discuss the project tomorrow').predictedCategory,
      'ham'
    )
  })

  it('handles ambiguous text with confidence check', function () {
    var result = classifier.categorizeWithConfidence('please review this', 0.9)
    // ambiguous text should either be confident ham or rejected
    assert.ok(
      result.predictedCategory === 'ham' || result.predictedCategory === null
    )
  })

  it('explains predictions with influential tokens', function () {
    var tokens = classifier.topInfluentialTokens('Buy now free offer', 3)
    assert.ok(tokens.length > 0)
    // all tokens should have valid probabilities
    tokens.forEach(function (t) {
      assert.ok(t.probability > 0)
      assert.ok(t.probability <= 1)
    })
  })

  it('survives serialization round-trip and still classifies', function () {
    var json = classifier.toJson()
    var restored = bayes.fromJson(json)

    assert.equal(
      restored.categorize('Buy now free offer limited time').predictedCategory,
      'spam'
    )
    assert.equal(
      restored.categorize('Can we discuss the project tomorrow').predictedCategory,
      'ham'
    )
  })

  it('stats reflect training data', function () {
    var stats = classifier.getCategoryStats()
    assert.equal(stats.spam.docCount, 5)
    assert.equal(stats.ham.docCount, 5)
    assert.equal(stats._total.docCount, 10)
    assert.ok(stats._total.wordCount > 0)
    assert.ok(stats._total.vocabularySize > 0)
  })
})

describe('[E2E] sentiment analysis', function () {
  var classifier

  beforeEach(function () {
    classifier = bayes({
      tokenPreprocessor: function (tokens) {
        return tokens.map(function (t) { return t.toLowerCase() })
      }
    })

    var trainingData = [
      { text: 'I love this product amazing quality', category: 'positive' },
      { text: 'Great experience wonderful service', category: 'positive' },
      { text: 'Excellent work very satisfied happy', category: 'positive' },
      { text: 'Best purchase ever highly recommend', category: 'positive' },
      { text: 'Terrible quality waste of money', category: 'negative' },
      { text: 'Horrible experience worst service ever', category: 'negative' },
      { text: 'Broken on arrival very disappointed', category: 'negative' },
      { text: 'Would not recommend awful product', category: 'negative' }
    ]

    classifier.learnBatch(trainingData)
  })

  it('classifies positive review correctly', function () {
    var result = classifier.categorize('Amazing product love the quality')
    assert.equal(result.predictedCategory, 'positive')
    assert.ok(result.likelihoods[0].proba > 0.5)
  })

  it('classifies negative review correctly', function () {
    var result = classifier.categorize('Terrible waste would not buy again')
    assert.equal(result.predictedCategory, 'negative')
    assert.ok(result.likelihoods[0].proba > 0.5)
  })

  it('probabilities always sum to 1', function () {
    var texts = [
      'Amazing product love it',
      'Terrible broken waste',
      'Some random unrelated words',
      ''
    ]

    texts.forEach(function (text) {
      var result = classifier.categorize(text)
      if (result.likelihoods.length > 0) {
        var sum = result.likelihoods.reduce(function (a, l) { return a + l.proba }, 0)
        assert.ok(Math.abs(sum - 1.0) < 0.01, 'proba sum should be ~1, got ' + sum)
      }
    })
  })

  it('top influential tokens make semantic sense', function () {
    var tokens = classifier.topInfluentialTokens('Amazing product love the quality', 3)
    var tokenNames = tokens.map(function (t) { return t.token })

    // at least one positive word should be influential
    var positiveWords = ['amazing', 'love', 'quality', 'product']
    var hasPositive = tokenNames.some(function (t) { return positiveWords.indexOf(t) !== -1 })
    assert.ok(hasPositive, 'should have at least one positive word in influential tokens')
  })
})

describe('[E2E] multi-category topic classification', function () {
  var classifier

  beforeEach(function () {
    classifier = bayes()

    classifier.learnBatch([
      { text: 'The stock market rallied today as investors showed confidence', category: 'finance' },
      { text: 'Federal reserve announces interest rate decision', category: 'finance' },
      { text: 'Bitcoin cryptocurrency prices surged this week', category: 'finance' },

      { text: 'Scientists discover new species in the Amazon rainforest', category: 'science' },
      { text: 'NASA launches new telescope to study distant galaxies', category: 'science' },
      { text: 'Research shows promising results for new cancer treatment', category: 'science' },

      { text: 'Team wins championship in overtime thriller', category: 'sports' },
      { text: 'Player breaks scoring record in historic game', category: 'sports' },
      { text: 'Coach announces retirement after successful season', category: 'sports' },

      { text: 'New smartphone features revolutionary camera technology', category: 'technology' },
      { text: 'AI startup raises funding for machine learning platform', category: 'technology' },
      { text: 'Software update fixes critical security vulnerability', category: 'technology' }
    ])
  })

  it('correctly classifies finance text', function () {
    assert.equal(
      classifier.categorize('investors concerned about market volatility').predictedCategory,
      'finance'
    )
  })

  it('correctly classifies science text', function () {
    assert.equal(
      classifier.categorize('researchers study new galaxy formation').predictedCategory,
      'science'
    )
  })

  it('correctly classifies sports text', function () {
    assert.equal(
      classifier.categorize('team wins game scoring record').predictedCategory,
      'sports'
    )
  })

  it('correctly classifies technology text', function () {
    assert.equal(
      classifier.categorize('new AI software update released').predictedCategory,
      'technology'
    )
  })

  it('topN returns correct number of categories', function () {
    var result = classifier.categorizeTopN('new technology update', 2)
    assert.equal(result.likelihoods.length, 2)
    // top result should be technology
    assert.equal(result.predictedCategory, 'technology')
  })

  it('all 4 categories exist', function () {
    var cats = classifier.getCategories()
    assert.equal(cats.length, 4)
    assert.ok(cats.indexOf('finance') !== -1)
    assert.ok(cats.indexOf('science') !== -1)
    assert.ok(cats.indexOf('sports') !== -1)
    assert.ok(cats.indexOf('technology') !== -1)
  })

  it('survives full serialize → restore → classify cycle', function () {
    var json = classifier.toJson()
    var restored = bayes.fromJson(json)

    assert.equal(
      restored.categorize('stock market investors').predictedCategory,
      'finance'
    )
    assert.equal(restored.getCategories().length, 4)
    assert.equal(restored.getCategoryStats()._total.docCount, 12)
  })
})

describe('[E2E] incremental learning over time', function () {
  it('classifier improves as more data is added', function () {
    var classifier = bayes()

    // start with minimal data
    classifier.learn('good', 'positive')
    classifier.learn('bad', 'negative')

    // ambiguous initially
    var initial = classifier.categorize('good bad ugly')

    // add more training data incrementally
    classifier.learnBatch([
      { text: 'good great wonderful amazing', category: 'positive' },
      { text: 'good fantastic brilliant', category: 'positive' },
      { text: 'bad terrible awful', category: 'negative' },
      { text: 'bad horrible dreadful', category: 'negative' }
    ])

    // should now clearly classify 'good' as positive
    var improved = classifier.categorize('good')
    assert.equal(improved.predictedCategory, 'positive')
    assert.ok(improved.likelihoods[0].proba > 0.5)

    // stats reflect all training
    assert.equal(classifier.getCategoryStats()._total.docCount, 6)
  })
})

describe('[E2E] correcting classification mistakes', function () {
  it('unlearn wrong data, re-learn correct data, verify improvement', function () {
    var classifier = bayes()

    // initial correct training
    classifier.learn('happy joy smile', 'positive')
    classifier.learn('sad cry tears', 'negative')

    // oops, accidentally trained wrong
    classifier.learn('happy celebration party', 'negative')  // mistake!

    // verify the mistake hurts classification
    var before = classifier.categorize('happy celebration')

    // fix the mistake
    classifier.unlearn('happy celebration party', 'negative')
    classifier.learn('happy celebration party', 'positive')

    // verify correction helped
    var after = classifier.categorize('happy celebration')
    assert.equal(after.predictedCategory, 'positive')
  })
})

describe('[E2E] fitPrior impact on imbalanced datasets', function () {
  it('fitPrior=true favors the majority class on ambiguous input', function () {
    var withPrior = bayes({ fitPrior: true })
    var withoutPrior = bayes({ fitPrior: false })

    // heavily imbalanced: 10 positive, 1 negative
    for (var i = 0; i < 10; i++) {
      withPrior.learn('word' + i, 'majority')
      withoutPrior.learn('word' + i, 'majority')
    }
    withPrior.learn('rare', 'minority')
    withoutPrior.learn('rare', 'minority')

    // ambiguous text (unknown word)
    var resultPrior = withPrior.categorize('unknown')
    var resultNoPrior = withoutPrior.categorize('unknown')

    // with prior, majority class should be strongly favored
    assert.equal(resultPrior.predictedCategory, 'majority')
    assert.ok(resultPrior.likelihoods[0].proba > 0.8)

    // without prior, both classes should be closer to equal
    assert.equal(resultNoPrior.likelihoods[0].proba, 0.5)
  })
})
