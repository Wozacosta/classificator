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

/**
 * Initializes a NaiveBayes instance from a JSON state representation.
 * Use this with classifier.toJson().
 *
 * @param  {String|Object} jsonStrOrObject   state representation obtained by classifier.toJson()
 * @return {NaiveBayes}                      Classifier
 */
module.exports.fromJson = jsonStrOrObject => {
  let parameters;

  try {
    switch (typeof jsonStrOrObject) {
      case 'string':
        parameters = JSON.parse(jsonStrOrObject);
        break;

      case 'object':
        parameters = jsonStrOrObject;
        break;

      default:
        throw new Error('');
    }
  } catch (e) {
    console.log(e);
    throw new Error('Naivebays.fromJson expects a valid JSON string or an object.')
  }

  // init a new classifier
  let classifier = new Naivebayes(parameters.options);

  // override the classifier's state
  STATE_KEYS.forEach(k => {
    if (!parameters[k]) {
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
const defaultTokenizer = text => {
  //remove punctuation from text - remove anything that isn't a word char or a space
  let rgxPunctuation = /[^(a-zA-ZA-Яa-я0-9_)+\s]/g;

  let sanitized = text.replace(rgxPunctuation, ' ');
  // tokens = tokens.filter(function(token) {
  //   return token.length >= _that.config.minimumLength;
  // });

  return sanitized.split(/\s+/);
};

/**
 * Naive-Bayes Classifier
 *
 * This is a naive-bayes classifier that uses Laplace Smoothing.
 *
 * Takes an (optional) options object containing:
 *   - `tokenizer`  => custom tokenization function
 *
 */
function Naivebayes(options) {
  // set options object
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

  //initialize our vocabulary and its size
  this.vocabulary = {};
  this.vocabularySize = 0;

  //number of documents we have learned from
  this.totalDocuments = 0;

  //document frequency table for each of our categories
  //=> for each category, how often were documents mapped to it
  this.docCount = {};

  //for each category, how many words total were mapped to it
  this.wordCount = {};

  //word frequency table for each category
  //=> for each category, how frequent was a given word mapped to it
  this.wordFrequencyCount = {};

  //hashmap of our category names
  this.categories = {};
}


/**
 * Initialize each of our data structure entries for this new category
 *
 * @param  {String} categoryName
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
 */
Naivebayes.prototype.removeCategory = function(categoryName) {
  if (!this.categories[categoryName]){
    return this;
  }
  //update the total number of documents we have learned from
  this.totalDocuments -= this.docCount[categoryName];

  Object.keys(this.wordFrequencyCount[categoryName]).forEach(token => {
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
 * train our naive-bayes classifier by telling it what `category`
 * the `text` corresponds to.
 *
 * @param  {String} text
 * @param  {String} category Category to learn as being text
 */
Naivebayes.prototype.learn = function(text, category) {
  //initialize category data structures if we've never seen this category
  this.initializeCategory(category);

  //update our count of how many documents mapped to this category
  this.docCount[category]++;

  //update the total number of documents we have learned from
  this.totalDocuments++;

  //normalize the text into a word array
  let tokens = this.tokenizer(text);

  //get a frequency count for each token in the text
  let frequencyTable = this.frequencyTable(tokens);

  Object.keys(frequencyTable).forEach(token => {

    let frequencyInText = frequencyTable[token];

    //add this word to our vocabulary if not already existing
    if (!this.vocabulary[token] || this.vocabulary[token] === 0) {
      this.vocabularySize++;
      this.vocabulary[token] = 1;
      // this.vocabulary[token] = frequencyInText;
    } else if (this.vocabulary[token] > 0) {
      this.vocabulary[token]++;
      // this.vocabulary[token] += frequencyInText;
    }


    //update the frequency information for this word in this category
    if (!this.wordFrequencyCount[category][token]) {
      this.wordFrequencyCount[category][token] = frequencyInText;
    }
    else this.wordFrequencyCount[category][token] += frequencyInText;

    //update the count of all words we have seen mapped to this category
    this.wordCount[category] += frequencyInText;
  });

  return this;
};

/**
 * untrain our naive-bayes classifier by telling it what `category`
 * the `text` to remove corresponds to.
 *
 * @param  {String} text
 * @param  {String} category Category to unlearn as being text
 */
Naivebayes.prototype.unlearn = function(text, category){
  //update our count of how many documents mapped to this category
  this.docCount[category]--;
  if (this.docCount[category] === 0){
    delete this.docCount[category];
  }

  //update the total number of documents we have learned from
  this.totalDocuments--;

  //normalize the text into a word array
  let tokens = this.tokenizer(text);

  //get a frequency count for each token in the text
  let frequencyTable = this.frequencyTable(tokens);

  /*
   Update our vocabulary and our word frequency count for this category
   */

  Object.keys(frequencyTable).forEach(token => {

    let frequencyInText = frequencyTable[token];

    //add this word to our vocabulary if not already existing
    if (this.vocabulary[token] && this.vocabulary[token] > 0) {
      this.vocabulary[token] -= frequencyInText;
      if (this.vocabulary[token] === 0) this.vocabularySize--;
    }


    this.wordFrequencyCount[category][token] -= frequencyInText;
    if (this.wordFrequencyCount[category][token] === 0){
      delete this.wordFrequencyCount[category][token];
    }

    //update the count of all words we have seen mapped to this category
    this.wordCount[category] -= frequencyInText;
    if (this.wordCount[category] === 0){
      delete this.wordCount[category];
      delete this.wordFrequencyCount[category];
    }
  });

  return this;
};


/**
 * Determine what category `text` belongs to.
 *
 * @param  {String} text
 *
 * @return {Object} The predicted category, and the likelihoods stats.
 */
Naivebayes.prototype.categorize = function(text){
  const tokens = this.tokenizer(text);
  const frequencyTable = this.frequencyTable(tokens);
  const categories = Object.keys(this.categories)
  const likelihoods = [];

  // iterate through our categories to find the one with max probability for this text
  categories.forEach(category => {
    //start by calculating the overall probability of this category
    //=>  out of all documents we've ever looked at, how many were
    //    mapped to this category
    let categoryLikelihood = this.docCount[category] / this.totalDocuments;

    //take the log to avoid underflow
    let logLikelihood = Math.log(categoryLikelihood);

    //now determine P( w | c ) for each word `w` in the text
    Object.keys(frequencyTable).forEach(token => {
      let termFrequencyInText = frequencyTable[token];
      let tokenProbability = this.tokenProbability(token, category);

      // determine the log of the P( w | c ) for this word
      logLikelihood += termFrequencyInText * Math.log(tokenProbability);
    });

    if (logLikelihood == Number.NEGATIVE_INFINITY) {
      console.log(`category ${category} had -Infinity odds`);
    }
    likelihoods.push({ category, logLikelihood });
  });

  const logsumexp = likelihoods => {
    let sum = 0
    likelihoods.forEach(likelihood => {
      sum += Math.exp(likelihood.logLikelihood)
    })
    return Math.log(sum)
  }

  const logProbX = logsumexp(likelihoods)
  likelihoods.forEach(likelihood => {
    likelihood.logProba = likelihood.logLikelihood - logProbX
    likelihood.proba = Math.exp(likelihood.logProba)
  })

  // sort to have first element with biggest probability
  likelihoods.sort((a, b) => b.proba - a.proba)

  return {
    likelihoods,
    predictedCategory: likelihoods[0].category
  }
};

/**
 * Calculate probability that a `token` belongs to a `category`
 *
 * @param  {String} token
 * @param  {String} category
 * @return {Number} probability
 */
Naivebayes.prototype.tokenProbability = function(token, category){
  //how many times this word has occurred in documents mapped to this category
  let wordFrequencyCount = this.wordFrequencyCount[category][token] || 0;

  //what is the count of all words that have ever been mapped to this category
  let wordCount = this.wordCount[category];


  //use laplace Add-1 Smoothing equation
  return (wordFrequencyCount + 1) / (wordCount + this.vocabularySize);
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
  let frequencyTable = Object.create(null);

  tokens.forEach(token => {
    if (!frequencyTable[token]) frequencyTable[token] = 1;
    else frequencyTable[token]++;
  });

  return frequencyTable;
};

/**
 * Dump the classifier's state as a JSON string.
 * @return {String} Representation of the classifier.
 */
Naivebayes.prototype.toJson = function() {
  let state = {};

  STATE_KEYS.forEach(k => (state[k] = this[k]));

  return JSON.stringify(state);
};
