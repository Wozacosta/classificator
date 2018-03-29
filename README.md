# classificator

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

Recommended: Node v6.0.0 +

```
npm install --save classificator
```


## Usage

```
const bayes = require('classificator')
const classifier = bayes()
```

### Teach your classifier

```
classifier.learn('amazing, awesome movie! Had a good time', 'positive')
classifier.learn('Buy my free viagra pill and get rich!', 'spam')
classifier.learn('I really hate dust and annoying cats', 'negative')
classifier.learn('LOL this sucks so hard', 'troll')
```

### Make your classifier unlearn

```
classifier.learn('i hate mornings', 'positive');
// uh oh, that was mistake. Time to unlearn
classifier.unlearn('i hate mornings', 'positive');
```

### Remove a category

```
classifier.removeCategory('troll');
```

###  categorization

```
classifier.categorize("I've always hated Martians");
// => {
        likelihoods: [
          {
            category: 'negative',
            logLikelihood: -17.241944258040537,
            logProba: -0.6196197927020783,
            proba: 0.538149006882628
          }, {
            category: 'positive',
            logLikelihood: -17.93509143860048,
            logProba: -1.312766973262022,
            proba: 0.26907450344131445
          }, {
            category: 'spam',
            logLikelihood: -18.26854831109384,
            logProba: -1.646223845755383,
            proba: 0.19277648967605832 }
        ],
        predictedCategory: 'negative'
      }
```

### serialize the classifier's state as a JSON string.

`let stateJson = classifier.toJson()`

### load the classifier back from its JSON representation.

`let revivedClassifier = bayes.fromJson(stateJson)`

note: `stateJson` can either be a JSON string (obtained from `classifier.toJson()`), or an object


--------


## API

### `let classifier = bayes([options])`

Returns an instance of a Naive-Bayes Classifier.

Pass in an optional `options` object to configure the instance. If you specify a `tokenizer` function in `options`, it will be used as the instance's tokenizer. It receives a (string) `text` argument - this is the string value that is passed in by you when you call `.learn()` or `.categorize()`. It must return an array of tokens. The default tokenizer removes punctuation and splits on spaces.

Eg.

```
let classifier = bayes({
    tokenizer: function (text) { return text.split(' ') }
})
```

### `classifier.learn(text, category)`

Teach your classifier what `category` should be associated with an array `text` of words.

### `classifier.unlearn(text, category)`

The classifier will unlearn the `text` that was associated with `category`.

### `classifier.removeCategory(category)`

The category is removed and the classifier data are updated accordingly.

### `classifier.categorize(text)`

*Parameters*

`text {String}`

*Returns*

`{Object}` An object with the `predictedCategory` and an array of the categories
ordered by likelihood (most likely first).

```
{
    likelihoods : [
      ...
      {
        category: 'positive',
        logLikelihood: -17.93509143860048,
        logProba: -1.312766973262022,
        proba: 0.26907450344131445
      },
      ...
    ],
    predictedCategory : 'negative'  //--> the main category bayes thinks text
                                          belongs to. As a string
}
```

### `classifier.toJson()`

Returns the JSON representation of a classifier.

### `let classifier = bayes.fromJson(jsonStr)`

Returns a classifier instance from the JSON representation. Use this with the JSON representation obtained from `classifier.toJson()`
