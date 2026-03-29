const Decimal = require('decimal.js').default;

/*
    Expose our naive-bayes generator function
 */

module.exports = function(options) {
  return new Naivebayes(options);
};

// keys we use to serialize a classifier's state
const STATE_KEYS = (module.exports.STATE_KEYS = [
  'categories',
  'docCount',
  'totalDocuments',
  'vocabulary',
  'vocabularySize',
  'wordCount',
  'wordFrequencyCount',
  'options',
]);
const DEFAULT_ALPHA = 1;
const DEFAULT_FIT_PRIOR = true;

/**
 * Initializes a NaiveBayes instance from a JSON state representation.
 * Use this with classifier.toJson().
 *
 * @param  {String|Object} jsonStrOrObject   state representation obtained by classifier.toJson()
 * @param  {Object}        [options]         optional options object (e.g. { tokenizer: fn })
 * @return {NaiveBayes}                      Classifier
 * @throws {Error} If input is not a valid JSON string or object, or if required state keys are missing.
 */
module.exports.fromJson = (jsonStrOrObject, options) => {
  let parameters;

  try {
    switch (typeof jsonStrOrObject) {
      case 'string':
        parameters = JSON.parse(jsonStrOrObject);
        break;

      case 'object':
        if (jsonStrOrObject === null) {
          throw new Error('');
        }
        parameters = jsonStrOrObject;
        break;

      default:
        throw new Error('');
    }
  } catch (e) {
    throw new Error('Naivebayes.fromJson expects a valid JSON string or an object.');
  }

  // merge any runtime-only options (e.g. tokenizer) into the restored options
  const restoredOptions = Object.assign({}, parameters.options, options);

  // init a new classifier
  const classifier = new Naivebayes(restoredOptions);

  // override the classifier's state
  STATE_KEYS.forEach((k) => {
    if (typeof parameters[k] === 'undefined') {
      throw new Error(
        `Naivebayes.fromJson: JSON string is missing an expected property: [${k}].`
      );
    }
    classifier[k] = parameters[k];
  });

  return classifier;
};

/**
 * Given an input string, tokenize it into an array of word tokens.
 * This is the default tokenization function used if user does not provide one in `options`.
 *
 * @param  {String} text
 * @return {Array}
 */
const defaultTokenizer = (text) => {
  const rgxPunctuation = /[^(a-zA-ZA-Яa-я0-9_)+\s]/g;
  const sanitized = text.replace(rgxPunctuation, ' ');
  return sanitized.split(/\s+/).filter(token => token.length > 0);
};

/**
 * Naive-Bayes Classifier
 *
 * This is a naive-bayes classifier that uses Laplace Smoothing.
 *
 * @param {Object}   [options]              Configuration options
 * @param {Function} [options.tokenizer]    Custom tokenization function. Receives a string,
 *                                          must return an array of string tokens.
 * @param {number}   [options.alpha=1]      Additive (Laplace) smoothing parameter.
 * @param {boolean}  [options.fitPrior=true] Whether to use learned prior probabilities.
 *                                          When false, uses uniform prior.
 * @throws {TypeError} If options is truthy but not a plain object.
 */
function Naivebayes(options) {
  this.options = {};
  if (typeof options !== 'undefined') {
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw TypeError(
        `NaiveBayes got invalid 'options': ${options}'. Pass in an object.`
      );
    }
    this.options = options;
  }

  this.tokenizer = this.options.tokenizer || defaultTokenizer;
  this.alpha = this.options.alpha === undefined ? DEFAULT_ALPHA : this.options.alpha;
  this.fitPrior = this.options.fitPrior === undefined ? DEFAULT_FIT_PRIOR : this.options.fitPrior;

  this.vocabulary = {};
  this.vocabularySize = 0;
  this.totalDocuments = 0;
  this.docCount = {};
  this.wordCount = {};
  this.wordFrequencyCount = {};
  this.categories = {};
}

/**
 * Initialize each of our data structure entries for this new category.
 *
 * @param  {String} categoryName
 * @return {Naivebayes} this
 */
Naivebayes.prototype.initializeCategory = function(categoryName) {
  if (!this.categories[categoryName]) {
    this.docCount[categoryName] = 0;
    this.wordCount[categoryName] = 0;
    this.wordFrequencyCount[categoryName] = {};
    this.categories[categoryName] = true;
  }
  return this;
};

/**
 * Properly remove a category, unlearning all words that were associated to it.
 *
 * @param  {String} categoryName
 * @return {Naivebayes} this
 */
Naivebayes.prototype.removeCategory = function(categoryName) {
  if (!this.categories[categoryName]) {
    return this;
  }
  this.totalDocuments -= this.docCount[categoryName];

  Object.keys(this.wordFrequencyCount[categoryName]).forEach((token) => {
    this.vocabulary[token]--;
    if (this.vocabulary[token] === 0) this.vocabularySize--;
  });

  delete this.docCount[categoryName];
  delete this.wordCount[categoryName];
  delete this.wordFrequencyCount[categoryName];
  delete this.categories[categoryName];

  return this;
};

/**
 * Train our naive-bayes classifier by telling it what `category`
 * the `text` corresponds to.
 *
 * @param  {String} text
 * @param  {String} category Category to learn as being text
 * @return {Naivebayes} this
 * @throws {TypeError} If text or category is not a string.
 */
Naivebayes.prototype.learn = function(text, category) {
  if (typeof text !== 'string') {
    throw new TypeError(`NaiveBayes: text must be a string, got ${typeof text}.`);
  }
  if (typeof category !== 'string') {
    throw new TypeError(`NaiveBayes: category must be a string, got ${typeof category}.`);
  }

  this.initializeCategory(category);

  this.docCount[category]++;
  this.totalDocuments++;

  const tokens = this.tokenizer(text);
  const frequencyTable = this.frequencyTable(tokens);

  Object.keys(frequencyTable).forEach((token) => {
    const frequencyInText = frequencyTable[token];

    if (!this.vocabulary[token] || this.vocabulary[token] === 0) {
      this.vocabularySize++;
      this.vocabulary[token] = 1;
    } else if (this.vocabulary[token] > 0) {
      this.vocabulary[token]++;
    }

    if (!this.wordFrequencyCount[category][token]) {
      this.wordFrequencyCount[category][token] = frequencyInText;
    } else this.wordFrequencyCount[category][token] += frequencyInText;

    this.wordCount[category] += frequencyInText;
  });

  return this;
};

/**
 * Untrain our naive-bayes classifier by telling it what `category`
 * the `text` to remove corresponds to.
 *
 * @param  {String} text
 * @param  {String} category Category to unlearn as being text
 * @return {Naivebayes} this
 * @throws {TypeError} If text or category is not a string.
 * @throws {Error} If category does not exist.
 */
Naivebayes.prototype.unlearn = function(text, category) {
  if (typeof text !== 'string') {
    throw new TypeError(`NaiveBayes: text must be a string, got ${typeof text}.`);
  }
  if (typeof category !== 'string') {
    throw new TypeError(`NaiveBayes: category must be a string, got ${typeof category}.`);
  }
  if (!this.categories[category]) {
    throw new Error(`NaiveBayes: cannot unlearn from non-existent category: '${category}'.`);
  }

  this.docCount[category]--;
  if (this.docCount[category] === 0) {
    delete this.docCount[category];
  }

  this.totalDocuments--;

  const tokens = this.tokenizer(text);
  const frequencyTable = this.frequencyTable(tokens);

  Object.keys(frequencyTable).forEach((token) => {
    const frequencyInText = frequencyTable[token];

    if (this.vocabulary[token] && this.vocabulary[token] > 0) {
      this.vocabulary[token]--;
      if (this.vocabulary[token] === 0) this.vocabularySize--;
    }

    this.wordFrequencyCount[category][token] -= frequencyInText;
    if (this.wordFrequencyCount[category][token] === 0) {
      delete this.wordFrequencyCount[category][token];
    }

    this.wordCount[category] -= frequencyInText;
    if (this.wordCount[category] === 0) {
      delete this.wordCount[category];
      delete this.wordFrequencyCount[category];
    }
  });

  // clean up category if no documents remain
  if (!this.docCount[category]) {
    delete this.categories[category];
  }

  return this;
};

/**
 * Determine what category `text` belongs to.
 *
 * @param  {String} text
 * @return {Object} The predicted category, and the likelihoods stats.
 * @throws {TypeError} If text is not a string.
 */
Naivebayes.prototype.categorize = function(text) {
  if (typeof text !== 'string') {
    throw new TypeError(`NaiveBayes: text must be a string, got ${typeof text}.`);
  }

  const tokens = this.tokenizer(text);
  const frequencyTable = this.frequencyTable(tokens);
  const categories = Object.keys(this.categories);
  const likelihoods = [];

  if (categories.length === 0) {
    return {
      likelihoods: [],
      predictedCategory: null
    };
  }

  categories.forEach((category) => {
    let categoryLikelihood;
    if (this.fitPrior) {
      categoryLikelihood = this.docCount[category] / this.totalDocuments;
    } else {
      categoryLikelihood = 1;
    }

    let logLikelihood = Decimal(categoryLikelihood);
    logLikelihood = logLikelihood.naturalLogarithm();

    Object.keys(frequencyTable).forEach((token) => {
      if (this.vocabulary[token] && this.vocabulary[token] > 0) {
        const termFrequencyInText = frequencyTable[token];
        const tokenProbability = this.tokenProbability(token, category);

        let logTokenProbability = Decimal(tokenProbability);
        logTokenProbability = logTokenProbability.naturalLogarithm();
        logLikelihood = logLikelihood.plus(termFrequencyInText * logTokenProbability);
      }
    });

    likelihoods.push({ category, logLikelihood });
  });

  const logsumexp = (likelihoods) => {
    let sum = new Decimal(0);
    likelihoods.forEach((likelihood) => {
      const x = Decimal(likelihood.logLikelihood);
      const a = Decimal.exp(x);
      sum = sum.plus(a);
    });

    return sum.naturalLogarithm();
  };

  const logProbX = logsumexp(likelihoods);
  likelihoods.forEach((likelihood) => {
    likelihood.logProba = Decimal(likelihood.logLikelihood).minus(logProbX);
    likelihood.proba = likelihood.logProba.naturalExponential();
    likelihood.logProba = likelihood.logProba.toNumber();
    likelihood.proba = likelihood.proba.toNumber();
    likelihood.logLikelihood = likelihood.logLikelihood.toNumber();
  });

  likelihoods.sort((a, b) => b.proba - a.proba);

  return {
    likelihoods,
    predictedCategory: likelihoods[0].category
  };
};

/**
 * Like categorize(), but returns only the top N most likely categories.
 *
 * @param  {String} text  The text to categorize.
 * @param  {number} n     Maximum number of categories to return.
 * @return {Object}       Same shape as categorize(), but with truncated likelihoods.
 */
Naivebayes.prototype.categorizeTopN = function(text, n) {
  const result = this.categorize(text);
  if (result.likelihoods.length > n) {
    result.likelihoods = result.likelihoods.slice(0, n);
  }
  return result;
};

/**
 * Calculate probability that a `token` belongs to a `category`.
 *
 * @param  {String} token
 * @param  {String} category
 * @return {Number} probability
 */
Naivebayes.prototype.tokenProbability = function(token, category) {
  const wordFrequencyCount = this.wordFrequencyCount[category][token] || 0;
  const wordCount = this.wordCount[category];

  // use laplace Add-1 Smoothing equation
  return (wordFrequencyCount + this.alpha) / (wordCount + this.alpha * this.vocabularySize);
};

/**
 * Build a frequency hashmap where
 * - the keys are the entries in `tokens`
 * - the values are the frequency of each entry in `tokens`
 *
 * @param  {Array} tokens  Normalized word array
 * @return {Object}
 */
Naivebayes.prototype.frequencyTable = function(tokens) {
  const frequencyTable = Object.create(null);

  tokens.forEach((token) => {
    if (!frequencyTable[token]) frequencyTable[token] = 1;
    else frequencyTable[token]++;
  });

  return frequencyTable;
};

/**
 * Dump the classifier's state as a JSON string.
 *
 * @return {String} Representation of the classifier.
 */
Naivebayes.prototype.toJson = function() {
  const state = {};
  STATE_KEYS.forEach(k => (state[k] = this[k]));
  return JSON.stringify(state);
};

/**
 * Get an array of all category names the classifier has learned.
 *
 * @return {String[]} Array of category name strings.
 */
Naivebayes.prototype.getCategories = function() {
  return Object.keys(this.categories);
};

/**
 * Learn from multiple text/category pairs at once.
 *
 * @param  {Array<{text: string, category: string}>} items  Array of training items.
 * @return {Naivebayes} this
 * @throws {TypeError} If items is not an array.
 */
Naivebayes.prototype.learnBatch = function(items) {
  if (!Array.isArray(items)) {
    throw new TypeError('NaiveBayes: learnBatch expects an array of { text, category } objects.');
  }
  items.forEach(item => {
    this.learn(item.text, item.category);
  });
  return this;
};

/**
 * Reset the classifier to its initial (untrained) state, preserving configuration options.
 *
 * @return {Naivebayes} this
 */
Naivebayes.prototype.reset = function() {
  this.vocabulary = {};
  this.vocabularySize = 0;
  this.totalDocuments = 0;
  this.docCount = {};
  this.wordCount = {};
  this.wordFrequencyCount = {};
  this.categories = {};
  return this;
};

/**
 * Get statistics about each category's training data.
 *
 * @return {Object} Map of category names to { docCount, wordCount, vocabularySize },
 *                  plus a _total key with aggregate stats.
 */
Naivebayes.prototype.getCategoryStats = function() {
  const stats = {};
  Object.keys(this.categories).forEach(category => {
    stats[category] = {
      docCount: this.docCount[category] || 0,
      wordCount: this.wordCount[category] || 0,
      vocabularySize: Object.keys(this.wordFrequencyCount[category] || {}).length
    };
  });
  stats._total = {
    docCount: this.totalDocuments,
    vocabularySize: this.vocabularySize
  };
  return stats;
};
