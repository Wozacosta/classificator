# classificator

[![NPM Licence shield](https://img.shields.io/github/license/Wozacosta/classificator.svg)](https://github.com/Wozacosta/classificator/blob/master/LICENSE)

Naive Bayes classifier for node.js

`bayes` takes a document (piece of text), and tells you what category that document belongs to.


_Forked from https://www.npmjs.com/package/bayes, and adds some functionnalities upon it (returning more informations when categorizing, unlearning)._

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

## Differences with bayes-proba (original package)

For now, apart from a less misleading package name, I also changed misnommers in categorizeObj 

~~probas~~ -> likelihoods
~~proba~~ -> logLikelihood
~~probaH~~ -> scaledLogLikelihood
~~chosenCategory~~ -> predictedCategory


## Usage

```
const bayes = require('bayes-probas')
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
classifier.categorize("I've always hated Martians"); // simple
// => 'negative'

classifier.categorize("I've always hated Martians", true); // verbose
// =>
// =>
{
    likelihoods: [
            {
                category: 'negative',
                logLikelihood: 0.008489, // log likelihood
                scaledLogLikelihood: 100 // log likelihood on a [0-100] scale, not probability (100 doesn't mean it's 100% certain)
            },
            {
                category: 'troll',
                logLikelihood: 0.00412,
                scaledLogLikelihood: 43
            },
            {
                category: 'spam',
                logLikelihood: 0.00152,
                scaledLogLikelihood: 18
            },
            {
                category: 'positive',
                logLikelihood: 0.000074,
                scaledLogLikelihood: 0
            },
      predictedCategory : 'negative'
}
```


### serialize the classifier's state as a JSON string.

`let stateJson = classifier.toJson()`

### load the classifier back from its JSON representation.

`let revivedClassifier = bayes.fromJson(stateJson)`

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

### `classifier.categorize(text, verbose)`

*Parameters*
  `text {String}`
  `verbose {Boolean}` whether or not it should returns more data associated with the categorization.

If not `verbose` :

    *Returns*
       `{String}` An object with the `category` it thinks `text` belongs to. Based on what it learned with `classifier.learn()`.
    ```
    {
      predictedCategory: 'positive'
    }
    ```

If `verbose`

    *Returns*
     `{Object}` An object with the `predictedCategory` and an array of the categories ordered by likelihood (most likely first).

    ```
    {
        likelihoods :  [
                        ...
                          {
                            category: 'spam',
                            logLikelihood: 0.0047591, // logarithmic likelihood
                            scaledLogLikelihood: 84 // likelihood on a scale from 0 to 100
                          },
                          ...
                       ]
        predictedCategory : 'negative'  //--> the main category bayes thinks the text belongs to. As a string
    }
    ```

### `classifier.toJson()`

Returns the JSON representation of a classifier.

### `let classifier = bayes.fromJson(jsonStr)`

Returns a classifier instance from the JSON representation. Use this with the JSON representation obtained from `classifier.toJson()`
