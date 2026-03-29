# classificator

[![CI](https://github.com/Wozacosta/classificator/actions/workflows/ci.yml/badge.svg)](https://github.com/Wozacosta/classificator/actions/workflows/ci.yml)
[![NPM Licence shield](https://img.shields.io/github/license/Wozacosta/classificator.svg)](https://github.com/Wozacosta/classificator/blob/master/LICENSE)
[![NPM release version shield](https://img.shields.io/npm/v/classificator.svg)](https://www.npmjs.com/package/classificator)

A fast, lightweight Naive Bayes classifier for Node.js with explainable predictions. Written in TypeScript with full type declarations. Ships dual CJS/ESM.

```
                    +-----------------+
   "great movie" -->|  classificator  |--> { predictedCategory: "positive", proba: 0.83 }
                    +-----------------+
                      |  trained on  |
                      |  your data   |
                      +--------------+
```


## What can I use this for?

You can use this for categorizing any text content into any arbitrary set of **categories**. For example:

- is an email **spam**, or **not spam** ?
- is a news article about **technology**, **politics**, or **sports** ?
- is a piece of text expressing **positive** emotions, or **negative** emotions?

```
                          +----------+
                     +--->| positive | 0.72
   "awesome movie"   |   +----------+
         |            |   +----------+
         v            +-->| negative | 0.18
   [ tokenize ]      |   +----------+
         |            |   +----------+
         v            +-->|  neutral | 0.10
   [ calculate ]------+   +----------+
   [ probability ]
```

More here: https://en.wikipedia.org/wiki/Naive_Bayes_classifier


## Installing

Recommended: Node v18.0.0 +

```
npm install classificator
```


## Quick Start

```ts
// ESM (recommended)
import bayes from 'classificator'

// or with named imports
import { Naivebayes, fromJson } from 'classificator'

// CJS (still works)
const bayes = require('classificator')
```

```ts
const classifier = bayes()

// Train
classifier.learn('amazing, awesome movie!', 'positive')
classifier.learn('terrible, boring film', 'negative')

// Classify
const result = classifier.categorize('awesome film')
console.log(result.predictedCategory) // => 'positive'
```

### TypeScript

Full type declarations are included. All interfaces are exported:

```ts
import bayes from 'classificator'
import type { CategorizeResult, NaivebayesOptions, Likelihood } from 'classificator'

const options: NaivebayesOptions = { alpha: 0.5, fitPrior: false }
const classifier = bayes(options)

classifier.learn('great movie', 'positive')
const result: CategorizeResult = classifier.categorize('great')
```


## How It Works

Classificator uses the [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) algorithm with Laplace smoothing. Here's the pipeline:

```
  Input Text
      |
      v
+-------------+     +------------------+     +-------------------+
|  Tokenizer  |---->|  Preprocessor    |---->|  Frequency Table  |
| split words |     | stopwords/stem   |     |  count each word  |
+-------------+     +------------------+     +-------------------+
                                                      |
              +---------------------------------------+
              |
              v
+---------------------------+     +------------------+
|  For each category:       |     |  Normalize with  |
|  P(cat) * P(w1|cat) *    |---->|  logsumexp for   |
|  P(w2|cat) * ...          |     |  final proba     |
+---------------------------+     +------------------+
                                          |
                                          v
                                  +------------------+
                                  |  Return sorted   |
                                  |  likelihoods +   |
                                  |  predictedCategory|
                                  +------------------+
```

**Laplace smoothing** prevents zero-probability issues — even words never seen in a category get a small probability instead of zeroing everything out.


## Usage

### Teach your classifier

```js
classifier.learn('amazing, awesome movie! Had a good time', 'positive')
classifier.learn('Buy my free viagra pill and get rich!', 'spam')
classifier.learn('I really hate dust and annoying cats', 'negative')
classifier.learn('LOL this sucks so hard', 'troll')
```

### Batch learning

```js
classifier.learnBatch([
  { text: 'amazing, awesome movie!', category: 'positive' },
  { text: 'Buy my free viagra pill', category: 'spam' },
  { text: 'I really hate dust', category: 'negative' }
])
```

### Make your classifier unlearn

```js
classifier.learn('i hate mornings', 'positive');
// uh oh, that was a mistake. Time to unlearn
classifier.unlearn('i hate mornings', 'positive');
```

If the last document in a category is unlearned, the category is automatically removed.

### Remove a category

```js
classifier.removeCategory('troll');
```

### Categorization

```js
classifier.categorize("I've always hated Martians");
// => {
//      likelihoods: [
//        { category: 'negative', proba: 0.538, logLikelihood: -17.24, logProba: -0.62 },
//        { category: 'positive', proba: 0.269, logLikelihood: -17.94, logProba: -1.31 },
//        { category: 'spam',     proba: 0.193, logLikelihood: -18.27, logProba: -1.65 }
//      ],
//      predictedCategory: 'negative'
//    }
```

### Categorize with confidence threshold

Reject low-confidence predictions instead of guessing:

```js
classifier.categorizeWithConfidence('some ambiguous text', 0.7);
// => predictedCategory is null if the top probability is below 0.7
//    likelihoods array is always returned in full
```

```
   "ambiguous text"
         |
         v
   [ categorize ]
         |
    proba = 0.42
         |
    0.42 < 0.70 ?  --yes-->  predictedCategory: null    (rejected)
         |
        no
         |
         v
    predictedCategory: "spam"   (accepted)
```

### Get top N categories

```js
classifier.categorizeTopN("I've always hated Martians", 2);
// => same as categorize(), but likelihoods array has at most 2 entries
```

### Understand why a prediction was made

```js
classifier.topInfluentialTokens("I've always hated Martians", 3);
// => [
//      { token: 'hated', probability: 0.42, frequency: 1 },
//      { token: 'always', probability: 0.21, frequency: 1 },
//      { token: 'Martians', probability: 0.12, frequency: 1 }
//    ]
```

```
  "I've always hated Martians"  -->  predicted: negative
                                          |
      Why?                                v
      +----------------------------------------------------+
      | Token     | P(token|negative) | Influence          |
      |-----------|-------------------|--------------------|
      | hated     | 0.42              | ################## |
      | always    | 0.21              | #########          |
      | Martians  | 0.12              | #####              |
      +----------------------------------------------------+
```

### Serialize / Deserialize

```js
// Save
let stateJson = classifier.toJson()

// Restore
let revivedClassifier = bayes.fromJson(stateJson)
```

`stateJson` can be a JSON string or a plain object.

**Important:** Functions (`tokenizer`, `tokenPreprocessor`) can't be serialized to JSON. Pass them back when restoring:

```js
let revivedClassifier = bayes.fromJson(stateJson, {
  tokenizer: myTokenizer,
  tokenPreprocessor: myPreprocessor
})
```

```
  Classifier                     JSON String                    Classifier
  (in memory)                    (on disk)                      (restored)
       |                              |                              |
       +--- toJson() --------------->|                              |
       |                              +--- fromJson(json, opts) --->|
       |                              |          ^                   |
       |    tokenizer: fn  -  LOST    |          |                   |
       |    alpha: 0.5     -  KEPT    |    pass functions            |
       |    fitPrior: true -  KEPT    |    back in opts              |
       |                              |                              |
```

### Inspect your classifier

```js
classifier.getCategories()
// => ['positive', 'spam', 'negative', 'troll']

classifier.getCategoryStats()
// => {
//      positive: { docCount: 1, wordCount: 7, vocabularySize: 7 },
//      spam:     { docCount: 1, wordCount: 8, vocabularySize: 8 },
//      ...
//      _total:   { docCount: 4, wordCount: 25, vocabularySize: 20 }
//    }
```

### Reset the classifier

```js
classifier.reset()
// clears all learned data but preserves options (tokenizer, alpha, fitPrior)
```

### Method chaining

Most methods return `this`, so you can chain calls:

```js
const result = bayes()
  .learn('happy fun', 'positive')
  .learn('sad bad', 'negative')
  .categorize('happy')
```


--------


## API

### `let classifier = bayes([options])`

Returns an instance of a Naive-Bayes Classifier.

| Option               | Type       | Default                    | Description                                                                                     |
|----------------------|------------|----------------------------|-------------------------------------------------------------------------------------------------|
| `tokenizer`          | `Function` | Splits on whitespace/punct | Custom tokenization function. Receives `text` (string), must return an array of string tokens.  |
| `tokenPreprocessor`  | `Function` | none                       | Transform tokens after tokenization (e.g. stopword removal, stemming, lowercasing). Receives and returns an array of tokens. |
| `alpha`              | `number`   | `1`                        | Additive (Laplace) smoothing parameter. Higher values = more conservative predictions. `0` disables smoothing (can cause zero-probability issues). |
| `fitPrior`           | `boolean`  | `true`                     | If `true`, prior probability is proportional to learned document frequencies (categories with more training docs are favored). If `false`, uses uniform prior (all categories equally likely before seeing the text). |

```js
let classifier = bayes({
    tokenizer: function (text) { return text.split(' ') },
    tokenPreprocessor: function (tokens) {
      var stopwords = new Set(['the', 'a', 'is', 'in'])
      return tokens
        .map(function (t) { return t.toLowerCase() })
        .filter(function (t) { return !stopwords.has(t) })
    },
    alpha: 0.5,
    fitPrior: false
})
```

#### Understanding `alpha` (Laplace smoothing)

```
  alpha controls how much probability "leaks" to unseen words:

  alpha = 0     Unseen words get 0 probability. Risky.
  alpha = 0.5   Lidstone smoothing. Less aggressive.
  alpha = 1     Standard Laplace. Good default.      <-- default
  alpha = 10    Very conservative. Small datasets.

  Effect on P(word|category):

         P(word|cat) = (count + alpha) / (total + alpha * vocabSize)
                        ──────────────   ─────────────────────────────
                        numerator gets    denominator grows with alpha
                        a boost           spreading probability to all
                                          possible words
```

#### Understanding `fitPrior`

```
  fitPrior: true (default)         fitPrior: false
  ─────────────────────────         ────────────────────────
  P(cat) = docCount / total         P(cat) = 1  (uniform)

  900 positive docs + 100 negative   Same data, but:
  P(positive) = 0.9                  P(positive) = P(negative)
  P(negative) = 0.1                  Only word content matters

  Good when training data            Good when training data
  reflects real-world                 is imbalanced but you want
  distribution                       fair comparison
```

### `classifier.learn(text, category)`

Teach your classifier what `category` should be associated with a `text` string.

Returns `this` for chaining. Throws `TypeError` if text or category is not a string.

### `classifier.learnBatch(items)`

Learn from multiple text/category pairs at once. `items` is an array of `{ text, category }` objects.

Returns `this` for chaining. Throws `TypeError` if items is not an array.

### `classifier.unlearn(text, category)`

The classifier will unlearn the `text` that was associated with `category`. If the last document in a category is unlearned, the category is automatically removed.

Returns `this` for chaining. Throws `Error` if the category does not exist.

### `classifier.removeCategory(category)`

The category is removed and the classifier data are updated accordingly. Vocabulary is cleaned up: tokens only present in the removed category are removed from the global vocabulary. No-op if the category does not exist.

Returns `this` for chaining.

### `classifier.categorize(text)`

Returns `{Object}` with `predictedCategory` and `likelihoods` array sorted by probability (highest first). Returns `{ predictedCategory: null, likelihoods: [] }` if no categories have been learned.

```js
{
    likelihoods: [
      { category: 'positive', logLikelihood: -17.94, logProba: -1.31, proba: 0.27 },
      ...
    ],
    predictedCategory: 'negative'
}
```

### `classifier.categorizeWithConfidence(text, threshold)`

Like `categorize()`, but sets `predictedCategory` to `null` if the top category's probability is below `threshold` (a number between 0 and 1). The `likelihoods` array is always returned in full. Throws `TypeError` if threshold is invalid.

### `classifier.categorizeTopN(text, n)`

Like `categorize()`, but returns only the top `n` most likely categories in the likelihoods array.

### `classifier.topInfluentialTokens(text[, n])`

Returns the top `n` (default 5) tokens that most influenced the predicted category, sorted by probability. Each entry has `{ token, probability, frequency }`.

### `classifier.getCategories()`

Returns an array of all category names the classifier has learned.

### `classifier.getCategoryStats()`

Returns an object with per-category stats (`docCount`, `wordCount`, `vocabularySize`) and a `_total` key with aggregate stats including total `wordCount`.

### `classifier.reset()`

Resets the classifier to its initial untrained state, preserving configuration options.

Returns `this` for chaining.

### `classifier.toJson()`

Returns the JSON representation of a classifier.

### `let classifier = bayes.fromJson(jsonStr[, options])`

Returns a classifier instance from the JSON representation. Use this with `classifier.toJson()`.

`jsonStr` can be a JSON string or a plain object.

`options` is an optional object for runtime-only options (e.g. `{ tokenizer: fn, tokenPreprocessor: fn }`) that cannot be serialized to JSON.


--------


## Typical Workflows

### Spam Filter

```
  +-----------+     +-----------+     +-------------+     +--------+
  | Collect   |---->| Train     |---->| Serialize   |---->| Deploy |
  | emails    |     | classifier|     | to JSON     |     | in app |
  +-----------+     +-----------+     +-------------+     +--------+
                         |                                     |
                    learn('buy now        fromJson(saved) then
                     free!!!', 'spam')    categorize(newEmail)
                    learn('meeting at
                     3pm', 'ham')
```

### Sentiment Analysis with Preprocessing

```js
const classifier = bayes({
  tokenPreprocessor: (tokens) => {
    const stops = new Set(['the', 'a', 'is', 'it', 'and', 'of', 'to'])
    return tokens
      .map(t => t.toLowerCase())
      .filter(t => !stops.has(t) && t.length > 2)
  }
})

// Train on labeled reviews
reviews.forEach(r => classifier.learn(r.text, r.sentiment))

// Classify new review
const result = classifier.categorize('This product is absolutely amazing!')
if (result.likelihoods[0].proba > 0.7) {
  console.log(`Confident: ${result.predictedCategory}`)
} else {
  console.log('Uncertain, needs human review')
}
```

### Model Persistence

```js
const fs = require('fs')

// Save trained model
fs.writeFileSync('model.json', classifier.toJson())

// Load later
const saved = fs.readFileSync('model.json', 'utf8')
const classifier = bayes.fromJson(saved, { tokenizer: myTokenizer })
```


--------


## Test Suite

The library includes a comprehensive test suite with **109 tests** (powered by Vitest):

```
  Unit tests (82)        - Individual method correctness, edge cases,
                           parameter validation, numerical stability

  Integration tests (7)  - Feature combinations: serialize/restore pipelines,
                           learn/unlearn/relearn cycles, preprocessor
                           consistency, method chaining workflows

  E2E tests (20)         - Real-world scenarios: spam detection, sentiment
                           analysis, multi-category topic classification,
                           incremental learning, mistake correction,
                           imbalanced dataset handling
```

Run with:

```
npm test
```


--------


## Changelog

### 1.0.0

**TypeScript rewrite:**
- Full TypeScript source with exported types (`NaivebayesOptions`, `CategorizeResult`, `Likelihood`, `InfluentialToken`, `CategoryStats`, `BatchItem`)
- Dual CJS/ESM output via tsup — `require()` and `import` both work
- Type declarations (`.d.ts`) included for TypeScript consumers
- ES6 class-based implementation (same API, better types)

**Modern tooling:**
- Build: tsup (esbuild-based, fast)
- Test: Vitest (replaces Mocha)
- CI: Node 18/20/22 with typecheck + build + test steps

**Breaking changes:**
- Minimum Node version raised to 18.0.0 (14 and 16 are EOL)
- Named ESM imports available: `import { Naivebayes, fromJson } from 'classificator'`

### 0.5.0

**New features:**
- `tokenPreprocessor` option for stopword removal, stemming, and custom token transforms
- `categorizeWithConfidence(text, threshold)` for rejecting low-confidence predictions
- `topInfluentialTokens(text, n)` for explainable classification
- `getCategories()`, `categorizeTopN()`, `learnBatch()`, `reset()`, `getCategoryStats()`
- Input validation on all public methods (throws TypeError for non-string inputs)

**Bug fixes:**
- Fixed `alpha: 0` being silently overridden to `1`
- Fixed `fromJson(null)` crash
- Fixed `unlearn()` not cleaning up categories when last document is removed
- Fixed `unlearn()` crash on non-existent category
- Fixed `categorize()` crash on empty classifier (now returns `predictedCategory: null`)
- Fixed default tokenizer returning empty tokens for empty strings
- Fixed `removeCategory()` not guarding against negative vocabulary counts
- Fixed `wordCount` going negative in `unlearn()` edge cases
- Fixed logsumexp numerical instability (now uses max-subtraction trick)
- Fixed `fromJson()` losing runtime options after state restoration
- Fixed error message typo and inconsistent capitalization

**Improvements:**
- Numerically stable logsumexp prevents underflow on large documents
- Tokenizer and tokenPreprocessor validation at construction time
- `getCategoryStats()` now includes `wordCount` in `_total`
- GitHub Actions CI for Node 14/16/18/20
- Comprehensive test suite (109 tests: unit + integration + E2E)
- Improved JSDoc and README documentation with diagrams

### 0.4.0

- Allow custom tokenizer to be passed to `fromJson()`

### 0.3.4

- Initial tracked version
