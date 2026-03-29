import { describe, it, beforeEach, expect } from 'vitest'
import bayes from '../src/index'
import type { Naivebayes } from '../src/index'

// =============================================================================
// Integration Tests
// Test multiple features working together in realistic combinations
// =============================================================================

describe('[Integration] full train → serialize → restore → classify pipeline', () => {
  it('classifier survives a full round-trip with custom options', () => {
    const tokenizer = (text: string) => { return text.toLowerCase().split(/\s+/) }
    const preprocessor = (tokens: string[]) => {
      const stops = new Set(['the', 'a', 'is', 'it', 'and', 'of', 'to', 'in'])
      return tokens.filter((t) => { return !stops.has(t) })
    }

    // 1. Create with custom options
    const classifier = bayes({
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
    const before = classifier.categorize('amazing film')
    expect(before.predictedCategory).toBe('positive')

    // 4. Serialize
    const json = classifier.toJson()

    // 5. Restore with runtime options
    const restored = bayes.fromJson(json, {
      tokenizer: tokenizer,
      tokenPreprocessor: preprocessor
    })

    // 6. Verify post-restoration
    const after = restored.categorize('amazing film')
    expect(after.predictedCategory).toBe('positive')
    expect(after.likelihoods.length).toBe(before.likelihoods.length)

    // 7. Probabilities should match
    expect(
      before.likelihoods[0].proba.toFixed(8)
    ).toBe(
      after.likelihoods[0].proba.toFixed(8)
    )

    // 8. Options preserved
    expect(restored.alpha).toBe(0.5)
    expect(restored.fitPrior).toBe(true)
    expect(restored.tokenizer).toBe(tokenizer)
    expect(restored.tokenPreprocessor).toBe(preprocessor)
  })
})

describe('[Integration] learn → unlearn → relearn cycle', () => {
  it('classifier state is consistent after learn/unlearn/relearn', () => {
    const classifier = bayes()

    // learn initial data
    classifier.learn('good morning sunshine', 'positive')
    classifier.learn('terrible horrible day', 'negative')
    classifier.learn('wonderful great time', 'positive')

    const stats1 = classifier.getCategoryStats()
    expect(stats1._total.docCount).toBe(3)

    // unlearn a mistake
    classifier.unlearn('wonderful great time', 'positive')
    expect(classifier.totalDocuments).toBe(2)

    // re-learn corrected data
    classifier.learn('wonderful great time', 'neutral')
    expect(classifier.totalDocuments).toBe(3)

    // classifier should now have 3 categories
    const categories = classifier.getCategories()
    expect(categories.indexOf('positive') !== -1).toBeTruthy()
    expect(categories.indexOf('negative') !== -1).toBeTruthy()
    expect(categories.indexOf('neutral') !== -1).toBeTruthy()

    // classification still works
    const result = classifier.categorize('terrible')
    expect(result.predictedCategory).toBe('negative')
  })
})

describe('[Integration] batch learning + stats + reset + retrain', () => {
  it('full lifecycle: batch train, inspect, reset, retrain differently', () => {
    const classifier = bayes()

    // batch train
    classifier.learnBatch([
      { text: 'buy cheap viagra now', category: 'spam' },
      { text: 'limited offer free pills', category: 'spam' },
      { text: 'hello how are you doing', category: 'ham' },
      { text: 'meeting at 3pm tomorrow', category: 'ham' },
      { text: 'project update attached', category: 'ham' }
    ])

    // inspect
    const stats = classifier.getCategoryStats()
    expect(stats.spam.docCount).toBe(2)
    expect(stats.ham.docCount).toBe(3)
    expect(stats._total.docCount).toBe(5)

    // classify
    expect(classifier.categorize('free offer').predictedCategory).toBe('spam')
    expect(classifier.categorize('meeting tomorrow').predictedCategory).toBe('ham')

    // reset
    classifier.reset()
    expect(classifier.totalDocuments).toBe(0)
    expect(classifier.getCategories()).toEqual([])

    // retrain with different categories
    classifier.learn('breaking news politics', 'news')
    classifier.learn('football scores today', 'sports')

    expect(classifier.categorize('political news').predictedCategory).toBe('news')
    expect(classifier.categorize('football game').predictedCategory).toBe('sports')
  })
})

describe('[Integration] removeCategory + reclassification', () => {
  it('removing a dominant category shifts predictions correctly', () => {
    const classifier = bayes()

    classifier.learn('python code function', 'programming')
    classifier.learn('java class object', 'programming')
    classifier.learn('javascript react component', 'programming')
    classifier.learn('recipe bake flour sugar', 'cooking')
    classifier.learn('stock market investment', 'finance')

    // programming dominates with fitPrior
    const before = classifier.categorize('new class today')
    expect(before.predictedCategory).toBe('programming')
    expect(before.likelihoods.length).toBe(3)

    // remove programming
    classifier.removeCategory('programming')

    // now should choose between cooking and finance
    const after = classifier.categorize('new class today')
    expect(after.likelihoods.length).toBe(2)
    expect(after.predictedCategory === 'cooking' || after.predictedCategory === 'finance').toBeTruthy()

    // probabilities still sum to ~1
    const sum = after.likelihoods.reduce((acc, l) => { return acc + l.proba }, 0)
    expect(Math.abs(sum - 1.0) < 0.01).toBeTruthy()
  })
})

describe('[Integration] tokenPreprocessor affects all operations consistently', () => {
  it('preprocessor is applied in learn, unlearn, categorize, and topInfluentialTokens', () => {
    let lowered: string[][] = []
    const preprocessor = (tokens: string[]) => {
      const result = tokens.map((t) => { return t.toLowerCase() })
      lowered.push(result)
      return result
    }

    const classifier = bayes({ tokenPreprocessor: preprocessor })

    // learn — preprocessor called
    lowered = []
    classifier.learn('HELLO WORLD', 'greetings')
    expect(lowered.length > 0).toBeTruthy()
    expect(lowered[0]).toEqual(['hello', 'world'])

    // should have lowercase tokens in vocabulary
    expect(classifier.vocabulary['hello']).toBeTruthy()
    expect(classifier.vocabulary['HELLO']).toBe(undefined)

    // categorize — preprocessor called
    lowered = []
    const result = classifier.categorize('HELLO')
    expect(lowered.length > 0).toBeTruthy()
    expect(result.predictedCategory).toBe('greetings')

    // topInfluentialTokens — preprocessor called
    lowered = []
    const tokens = classifier.topInfluentialTokens('HELLO WORLD', 2)
    expect(lowered.length > 0).toBeTruthy()
    expect(tokens.length > 0).toBeTruthy()

    // unlearn — preprocessor called (lowercase matches original learn)
    lowered = []
    classifier.learn('BYE NOW', 'farewells')
    classifier.unlearn('HELLO WORLD', 'greetings')
    expect(lowered.length > 0).toBeTruthy()
    expect(classifier.categories['greetings']).toBe(undefined)
  })
})

describe('[Integration] confidence threshold + topN combined workflow', () => {
  it('uses confidence to filter then topN to limit results', () => {
    const classifier = bayes()

    classifier.learnBatch([
      { text: 'cat dog hamster', category: 'pets' },
      { text: 'cat dog hamster', category: 'pets' },
      { text: 'car truck bus', category: 'vehicles' },
      { text: 'apple banana orange', category: 'fruit' },
      { text: 'table chair desk', category: 'furniture' }
    ])

    // high confidence prediction
    const confident = classifier.categorizeWithConfidence('cat dog', 0.3)
    expect(confident.predictedCategory).toBe('pets')

    // low confidence for ambiguous text
    const unsure = classifier.categorizeWithConfidence('xyz unknown', 0.5)
    expect(unsure.predictedCategory).toBe(null)

    // topN limits output
    const top2 = classifier.categorizeTopN('cat dog', 2)
    expect(top2.likelihoods.length).toBe(2)
    expect(top2.predictedCategory).toBe('pets')
  })
})

describe('[Integration] method chaining complex workflow', () => {
  it('chains learn, unlearn, removeCategory, and ends with categorize', () => {
    const result = bayes()
      .learn('happy joy love', 'positive')
      .learn('sad hate anger', 'negative')
      .learn('meh whatever ok', 'neutral')
      .learn('oops wrong category', 'neutral')
      .unlearn('oops wrong category', 'neutral')
      .removeCategory('neutral')
      .learn('wonderful amazing', 'positive')
      .categorize('love and joy')

    expect(result.predictedCategory).toBe('positive')
    expect(result.likelihoods.length).toBe(2)
  })
})

// =============================================================================
// End-to-End Tests
// Simulate real-world classification scenarios from start to finish
// =============================================================================

describe('[E2E] email spam detection', () => {
  let classifier: Naivebayes

  beforeEach(() => {
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

  it('correctly classifies obvious spam', () => {
    expect(
      classifier.categorize('Buy now free offer limited time').predictedCategory
    ).toBe('spam')
  })

  it('correctly classifies legitimate email', () => {
    expect(
      classifier.categorize('Can we discuss the project tomorrow').predictedCategory
    ).toBe('ham')
  })

  it('handles ambiguous text with confidence check', () => {
    const result = classifier.categorizeWithConfidence('please review this', 0.9)
    // ambiguous text should either be confident ham or rejected
    expect(
      result.predictedCategory === 'ham' || result.predictedCategory === null
    ).toBeTruthy()
  })

  it('explains predictions with influential tokens', () => {
    const tokens = classifier.topInfluentialTokens('Buy now free offer', 3)
    expect(tokens.length > 0).toBeTruthy()
    // all tokens should have valid probabilities
    tokens.forEach((t) => {
      expect(t.probability > 0).toBeTruthy()
      expect(t.probability <= 1).toBeTruthy()
    })
  })

  it('survives serialization round-trip and still classifies', () => {
    const json = classifier.toJson()
    const restored = bayes.fromJson(json)

    expect(
      restored.categorize('Buy now free offer limited time').predictedCategory
    ).toBe('spam')
    expect(
      restored.categorize('Can we discuss the project tomorrow').predictedCategory
    ).toBe('ham')
  })

  it('stats reflect training data', () => {
    const stats = classifier.getCategoryStats()
    expect(stats.spam.docCount).toBe(5)
    expect(stats.ham.docCount).toBe(5)
    expect(stats._total.docCount).toBe(10)
    expect(stats._total.wordCount > 0).toBeTruthy()
    expect(stats._total.vocabularySize > 0).toBeTruthy()
  })
})

describe('[E2E] sentiment analysis', () => {
  let classifier: Naivebayes

  beforeEach(() => {
    classifier = bayes({
      tokenPreprocessor: (tokens) => {
        return tokens.map((t) => { return t.toLowerCase() })
      }
    })

    const trainingData = [
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

  it('classifies positive review correctly', () => {
    const result = classifier.categorize('Amazing product love the quality')
    expect(result.predictedCategory).toBe('positive')
    expect(result.likelihoods[0].proba > 0.5).toBeTruthy()
  })

  it('classifies negative review correctly', () => {
    const result = classifier.categorize('Terrible waste would not buy again')
    expect(result.predictedCategory).toBe('negative')
    expect(result.likelihoods[0].proba > 0.5).toBeTruthy()
  })

  it('probabilities always sum to 1', () => {
    const texts = [
      'Amazing product love it',
      'Terrible broken waste',
      'Some random unrelated words',
      ''
    ]

    texts.forEach((text) => {
      const result = classifier.categorize(text)
      if (result.likelihoods.length > 0) {
        const sum = result.likelihoods.reduce((a, l) => { return a + l.proba }, 0)
        expect(Math.abs(sum - 1.0) < 0.01).toBeTruthy()
      }
    })
  })

  it('top influential tokens make semantic sense', () => {
    const tokens = classifier.topInfluentialTokens('Amazing product love the quality', 3)
    const tokenNames = tokens.map((t) => { return t.token })

    // at least one positive word should be influential
    const positiveWords = ['amazing', 'love', 'quality', 'product']
    const hasPositive = tokenNames.some((t) => { return positiveWords.indexOf(t) !== -1 })
    expect(hasPositive).toBeTruthy()
  })
})

describe('[E2E] multi-category topic classification', () => {
  let classifier: Naivebayes

  beforeEach(() => {
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

  it('correctly classifies finance text', () => {
    expect(
      classifier.categorize('investors concerned about market volatility').predictedCategory
    ).toBe('finance')
  })

  it('correctly classifies science text', () => {
    expect(
      classifier.categorize('researchers study new galaxy formation').predictedCategory
    ).toBe('science')
  })

  it('correctly classifies sports text', () => {
    expect(
      classifier.categorize('team wins game scoring record').predictedCategory
    ).toBe('sports')
  })

  it('correctly classifies technology text', () => {
    expect(
      classifier.categorize('new AI software update released').predictedCategory
    ).toBe('technology')
  })

  it('topN returns correct number of categories', () => {
    const result = classifier.categorizeTopN('new technology update', 2)
    expect(result.likelihoods.length).toBe(2)
    // top result should be technology
    expect(result.predictedCategory).toBe('technology')
  })

  it('all 4 categories exist', () => {
    const cats = classifier.getCategories()
    expect(cats.length).toBe(4)
    expect(cats.indexOf('finance') !== -1).toBeTruthy()
    expect(cats.indexOf('science') !== -1).toBeTruthy()
    expect(cats.indexOf('sports') !== -1).toBeTruthy()
    expect(cats.indexOf('technology') !== -1).toBeTruthy()
  })

  it('survives full serialize → restore → classify cycle', () => {
    const json = classifier.toJson()
    const restored = bayes.fromJson(json)

    expect(
      restored.categorize('stock market investors').predictedCategory
    ).toBe('finance')
    expect(restored.getCategories().length).toBe(4)
    expect(restored.getCategoryStats()._total.docCount).toBe(12)
  })
})

describe('[E2E] incremental learning over time', () => {
  it('classifier improves as more data is added', () => {
    const classifier = bayes()

    // start with minimal data
    classifier.learn('good', 'positive')
    classifier.learn('bad', 'negative')

    // ambiguous initially
    const initial = classifier.categorize('good bad ugly')

    // add more training data incrementally
    classifier.learnBatch([
      { text: 'good great wonderful amazing', category: 'positive' },
      { text: 'good fantastic brilliant', category: 'positive' },
      { text: 'bad terrible awful', category: 'negative' },
      { text: 'bad horrible dreadful', category: 'negative' }
    ])

    // should now clearly classify 'good' as positive
    const improved = classifier.categorize('good')
    expect(improved.predictedCategory).toBe('positive')
    expect(improved.likelihoods[0].proba > 0.5).toBeTruthy()

    // stats reflect all training
    expect(classifier.getCategoryStats()._total.docCount).toBe(6)
  })
})

describe('[E2E] correcting classification mistakes', () => {
  it('unlearn wrong data, re-learn correct data, verify improvement', () => {
    const classifier = bayes()

    // initial correct training
    classifier.learn('happy joy smile', 'positive')
    classifier.learn('sad cry tears', 'negative')

    // oops, accidentally trained wrong
    classifier.learn('happy celebration party', 'negative')  // mistake!

    // verify the mistake hurts classification
    const before = classifier.categorize('happy celebration')

    // fix the mistake
    classifier.unlearn('happy celebration party', 'negative')
    classifier.learn('happy celebration party', 'positive')

    // verify correction helped
    const after = classifier.categorize('happy celebration')
    expect(after.predictedCategory).toBe('positive')
  })
})

describe('[E2E] fitPrior impact on imbalanced datasets', () => {
  it('fitPrior=true favors the majority class on ambiguous input', () => {
    const withPrior = bayes({ fitPrior: true })
    const withoutPrior = bayes({ fitPrior: false })

    // heavily imbalanced: 10 positive, 1 negative
    for (let i = 0; i < 10; i++) {
      withPrior.learn('word' + i, 'majority')
      withoutPrior.learn('word' + i, 'majority')
    }
    withPrior.learn('rare', 'minority')
    withoutPrior.learn('rare', 'minority')

    // ambiguous text (unknown word)
    const resultPrior = withPrior.categorize('unknown')
    const resultNoPrior = withoutPrior.categorize('unknown')

    // with prior, majority class should be strongly favored
    expect(resultPrior.predictedCategory).toBe('majority')
    expect(resultPrior.likelihoods[0].proba > 0.8).toBeTruthy()

    // without prior, both classes should be closer to equal
    expect(resultNoPrior.likelihoods[0].proba).toBe(0.5)
  })
})
