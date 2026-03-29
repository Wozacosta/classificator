# classificator

[![CI](https://github.com/Wozacosta/classificator/actions/workflows/ci.yml/badge.svg)](https://github.com/Wozacosta/classificator/actions/workflows/ci.yml)
[![NPM Licence shield](https://img.shields.io/github/license/Wozacosta/classificator.svg)](https://github.com/Wozacosta/classificator/blob/master/LICENSE)
[![NPM release version shield](https://img.shields.io/npm/v/classificator.svg)](https://www.npmjs.com/package/classificator)

Naive Bayes classifier for node.js

`bayes` takes a document (piece of text), and tells you what category that document belongs to.


## What can I use this for?

You can use this for categorizing any text content into any arbitrary set of **categories**. For example:

- is an email **spam**, or **not spam** ?
- is a news article about **technology**, **politics**, or **sports** ?
- is a piece of text expressing **positive** emotions, or **negative** emotions?

More here: https://en.wikipedia.org/wiki/Naive_Bayes_classifier


## Installing

Recommended: Node v14.0.0 +

```
npm install --save classificator
```


## Usage

```js
const bayes = require('classificator')
const classifier = bayes()
```

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

### Remove a category

```js
classifier.removeCategory('troll');
```

### Categorization

```js
classifier.categorize("I've always hated Martians");
// => {
//      likelihoods: [
//        {
//          category: 'negative',
//          logLikelihood: -17.241944258040537,
//          logProba: -0.6196197927020783,
//          proba: 0.538149006882628
//        }, {
//          category: 'positive',
//          logLikelihood: -17.93509143860048,
//          logProba: -1.312766973262022,
//          proba: 0.26907450344131445
//        }, {
//          category: 'spam',
//          logLikelihood: -18.26854831109384,
//          logProba: -1.646223845755383,
//          proba: 0.19277648967605832
//        }
//      ],
//      predictedCategory: 'negative'
//    }
```

### Get top N categories

```js
classifier.categorizeTopN("I've always hated Martians", 2);
// => same as categorize(), but likelihoods array has at most 2 entries
```

### Serialize the classifier's state as a JSON string

```js
let stateJson = classifier.toJson()
```

### Load the classifier back from its JSON representation

```js
let revivedClassifier = bayes.fromJson(stateJson)
```

Note: `stateJson` can either be a JSON string (obtained from `classifier.toJson()`), or an object.

You can pass runtime options (like a custom tokenizer) that cannot be serialized to JSON:

```js
let revivedClassifier = bayes.fromJson(stateJson, { tokenizer: myTokenizer })
```

### Inspect your classifier

```js
classifier.getCategories()
// => ['positive', 'spam', 'negative', 'troll']

classifier.getCategoryStats()
// => {
//      positive: { docCount: 1, wordCount: 7, vocabularySize: 7 },
//      spam: { docCount: 1, wordCount: 8, vocabularySize: 8 },
//      ...
//      _total: { docCount: 4, vocabularySize: 25 }
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

Pass in an optional `options` object to configure the instance.

| Option      | Type       | Default                    | Description                                                                                     |
|-------------|------------|----------------------------|-------------------------------------------------------------------------------------------------|
| `tokenizer` | `Function` | Splits on whitespace/punct | Custom tokenization function. Receives `text` (string), must return an array of string tokens.  |
| `alpha`     | `number`   | `1`                        | Additive (Laplace) smoothing parameter. Set to `0` to disable smoothing.                        |
| `fitPrior`  | `boolean`  | `true`                     | If `true`, uses learned document frequency as prior. If `false`, uses uniform prior.             |

```js
let classifier = bayes({
    tokenizer: function (text) { return text.split(' ') },
    alpha: 0.5,
    fitPrior: false
})
```

### `classifier.learn(text, category)`

Teach your classifier what `category` should be associated with a `text` string.

Returns `this` for chaining.

### `classifier.learnBatch(items)`

Learn from multiple text/category pairs at once. `items` is an array of `{ text, category }` objects.

Returns `this` for chaining.

### `classifier.unlearn(text, category)`

The classifier will unlearn the `text` that was associated with `category`. Throws if the category does not exist.

Returns `this` for chaining.

### `classifier.removeCategory(category)`

The category is removed and the classifier data are updated accordingly. No-op if the category does not exist.

Returns `this` for chaining.

### `classifier.categorize(text)`

*Parameters*

`text {String}`

*Returns*

`{Object}` An object with the `predictedCategory` and an array of the categories
ordered by likelihood (most likely first). Returns `{ predictedCategory: null, likelihoods: [] }` if no categories have been learned.

```js
{
    likelihoods: [
      {
        category: 'positive',
        logLikelihood: -17.93509143860048,
        logProba: -1.312766973262022,
        proba: 0.26907450344131445
      },
      ...
    ],
    predictedCategory: 'negative'
}
```

### `classifier.categorizeTopN(text, n)`

Like `categorize()`, but returns only the top `n` most likely categories in the likelihoods array.

### `classifier.getCategories()`

Returns an array of all category names the classifier has learned.

### `classifier.getCategoryStats()`

Returns an object with per-category stats (`docCount`, `wordCount`, `vocabularySize`) and a `_total` key with aggregate stats.

### `classifier.reset()`

Resets the classifier to its initial untrained state, preserving configuration options (tokenizer, alpha, fitPrior).

Returns `this` for chaining.

### `classifier.toJson()`

Returns the JSON representation of a classifier.

### `let classifier = bayes.fromJson(jsonStr[, options])`

Returns a classifier instance from the JSON representation. Use this with the JSON representation obtained from `classifier.toJson()`.

`jsonStr` can be a JSON string or a plain object.

`options` is an optional object for runtime-only options (e.g. `{ tokenizer: fn }`) that cannot be serialized to JSON.
