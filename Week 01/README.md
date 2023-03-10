## 1. What is Natural Language Processing?

<img align="right" width="400" src="https://user-images.githubusercontent.com/40186859/220372533-59c81502-85ac-49d5-947a-13341d080f30.png" />

Natural language processing is a multidisciplinary field that combines techniques from computer science,  linguistics, and artificial intelligence to develop algorithms and models that enable machines to understand, analyze, and generate human language, such as text, speech, and even gestures.

For example:
- One application of natural language processing is virtual assistants like Apple's Siri or Amazon's Alexa, which can understand spoken commands and respond to them in natural language. 
- Another example is sentiment analysis, which uses machine learning to analyze social media posts and identify the sentiment or emotion behind them. This technology can help companies understand how people feel about their products or services, and improve their marketing strategies accordingly.
- Here's a funny example of NLP

  Q: Why did the computer go to the doctor?

  A: Because it had a virus! The doctor used natural language processing to diagnose the issue and prescribed some anti-virus software to help the computer get better.

  Of course, this is just a joke, but it highlights the idea that natural language processing can be used to analyze and understand human language, even when it's used in a humorous context. In reality, NLP has many practical applications in fields like healthcare, finance, and customer service, but it's always fun to imagine what kind of conversations machines could have if they were truly fluent in natural language.

### 1.1. Working with Text Data

Text data poses unique challenges and requires distinct solutions compared to other types of datasets. Preprocessing and cleaning of text data is often more intensive than with other data formats, in order to prepare it for statistical analysis or machine learning.

### 1.2. What is vectorization?

- Vectorization" is a term used in the context of converting input data, such as text, into numerical vectors to make it compatible with machine learning models. This approach has been in use since the advent of computers and has proven to be highly effective across various fields, including natural language processing

- In Machine Learning, vectorization is a step in feature extraction. The idea is to get some distinct features out of the text for the model to train on, by converting text to numerical vectors.

Points to be remember:

Most simple of all the simple techniques involves three operation

- Tokenization: First, the input text is tokenized. A sentence is represented as a list of its constituent words, and it???s done for all the input sentences.

- Vocabulary creation: Of all the obtained tokenized words, only unique words are selected to create the vocabulary and then sorted by alphabetical order.

- Vector Creation: Finally, a sparse matrix is created for the input, out of the frequency of vocabulary words. In this sparse matrix, each row is a sentence vector whose length (the columns of the matrix) is equal to the size of the vocabulary.


#### 1.1.1. Bag of Words

<img align="right" width="400" src="https://user-images.githubusercontent.com/40186859/220844654-f3f931e5-a03f-4bfe-a5cd-55211510a505.png" />

The most common approach to working with text involves vectorizing it by creating a Bag of Words, which accurately describes the final product as containing information about all important words in the text individually, but not in any particular order. This process involves throwing every word in a corpus into a bag, which, with a large enough corpus, reveals certain patterns that may emerge. For example, a bag of words made from Shakespeare's Hamlet is likely more similar to a bag of words made from Macbeth than to something like The Hunger Games. The simplest way to create a bag of words is to count how many times each unique word is used in a corpus, and having a number for every word enables us to treat each bag as a vector, thereby opening up all kinds of machine learning tools for use

Now let's explore bag of word in more details.

The CountVectorizer from Scikit-learn library is commonly used for creating a bag of words representation of text. The CountVectorizer converts a collection of text documents into a matrix of token counts, where each row represents a document, each column represents a word, and the values in the matrix are the frequency counts of each word in the corresponding document. This approach to text vectorization is useful for various machine learning tasks such as classification, clustering, and information retrieval.

It's natural to feel curious about how sentiment analysis works. When it comes to analyzing textual data for sentiment, we can't simply fit the raw text into a model. First, we need to convert the text into a numerical format using vectors, which is a common approach in NLP.

Suppose I have some positive examples over here:

**Sentence 01**: He is an intelligent boy

**Sentence 02**: She is an intelligent girl

**Sentence 03**: Both boy and girl are an intelligent 

During text preprocessing, there are a few tasks we need to perform. First, we need to convert the sentence to lowercase. Next, we should remove any stop words, which are commonly used words that don't carry much meaning, such as "the", "and", or "is". Finally, we can apply stemming or lemmatization, which involves reducing words to their root form, in order to further standardize the text.

After applying these steps, the resulting sentence would be transformed into a cleaner and more uniform representation that is more suitable for analysis or modeling.

**Sentence 01**: intelligent boy

**Sentence 02**: intelligent girl

**Sentence 03**: boy girl intelligent 

Text processing is the initial step, but our main focus is on how to derive vectors using bag of words. ????

To achieve this, we need to analyze each word in the pre-processed sentence and determine its frequency. By doing so, we can create a representation of the text in the form of a vector

| Words | Frequency |
|------ | :----------: |
| Intelligent | 3 |
| boy | 2 |
| girl | 2 |

Note: When calculating word frequency, the words may not be in order, but it's essential to sort them in descending order to make it easier to analyze the most important words in the text.

Now let's appy `Bag of Words`

|  | $f_1$| $f_2$ | $f_3$ |
|------ |------ | ---------- | ----------|
|  | intelligent | boy | girl|
| vector of sentence 01 | 1 | 1 | 0 |
| vector of sentence 02 | 1 | 0 | 1 |
| vector of sentence 03 | 1 | 1 | 1 |

Finally we have derived vectors using bag of words.???? While bag of words is a useful technique for text analysis, it also has its disadvantages.  One of the main disadvantages is that it doesn't take into account the context in which words appear. As a result, words with multiple meanings can be assigned the same vector representation, leading to ambiguity in the analysis. Additionally, bag of words doesn't capture the relationship between words, such as synonyms and antonyms, which can affect the accuracy of natural language processing tasks.

<b> Listed drawback of using a bag of words </b>

- If the new sentences contain new words, then our vocabulary size would increase and thereby, the length of vector would increase too.
- Additionally, the vectors would also contain many 0s, thereby resulting in a sparse matrix (which is what we would like to avoid)
- We are retaining no information on the grammar of the sentences nor the ordering of the words in the text.

From the above-generated vectors, it can be observed that the values assigned to the vectors are either 1 or 0. However, an important point to note is that both "intelligent" and "boy" have been assigned a value of 1, despite having different semantic meanings. This makes it difficult to determine which word holds greater importance. In sentiment analysis, it is crucial to identify the words that carry more weightage in determining the sentiment.

To overcome some of the limitations of the bag-of-words model we have something call as `TF-IDF` which is also called `Term Frequency and inverse document frequency` ???? 

Term Frequency - Inverse Document Frequency (TF-IDF) is a numerical statistics that is intended to reflect how important a word is to a document in a collection or corpus. 

Term Frequency (TF) - If is a measure of how frequently a term, t appears in a document d. 

$$t_{f_{t_i}d} = \frac{n_{t_id}}{Number\ of\ terms\ in\ the\ document}$$

Here, in the numerator, n is the number of times the term 't' appears in the document 'd'. Thus, each document and term would have its own TF value.

Let's take an example:

- This movie is very scary and long
- This movie is not scary and is slow
- This movie is spooky and good

First we will build a vocabulary from all unique words in above three movie reviews. The vocabulary consists of these 11 words.

Vocabulary: 'This', 'movie, 'is', 'very', 'scary', 'and', 'long', 'not, 'slow', 'spooky', 'good' 

- Number of words in first example: 7
- Number of words in second example: 8
- Number of word in third example: 6 


Example:

TF of the word `this` in second sentence: = $\frac{number\ of\ times\ this\ appear\ in\ second\ sentence}{number\ of\ terms\ in\ second\ sentence}$ = $\frac{1}{8}$

We can calculate the term frequencies for all terms and all the sentence in the number

| Term | Sentence 1 | Sentence 2 | Sentence 3 | TF Sentence 1 | TF Sentence 2 | TF Sentence 3|
| ---- | ---| ------- | -------  |-------  | -------  | ------- |
| This | 1 | 1 | 1 | $\frac{1}{7}$ | $\frac{1}{8}$ | $\frac{1}{6}$ |
| movie | 1 | 1 | 1 | $\frac{1}{7}$ | $\frac{1}{8}$ | $\frac{1}{6}$ |
| is | 1 | 2 | 1 | $\frac{1}{7}$ | $\frac{1}{4}$ | $\frac{1}{6}$ |
| very | 1 | 0 | 0 | $\frac{1}{7}$ | 0 | 0 |
| scary | 1 | 1 | 0 | $\frac{1}{7}$ | $\frac{1}{8}$ | 0 |
| and | 1 | 1 | 1 | $\frac{1}{7}$ | $\frac{1}{8}$ | $\frac{1}{6}$ |
| long | 1 | 0 | 0 | $\frac{1}{7}$ | 0 | 0 |
| not | 0 | 1 | 0 | 0 | $\frac{1}{8}$ | 0|
| slow | 0 | 1 | 0 | 0 | $\frac{1}{8}$ | 0|
| spooky | 0 | 0 | 1 | 0 | 0 | $\frac{1}{6}$|
| good | 0 | 0 | 1 | 0 | 0 | $\frac{1}{6}$|

Inverse Document Frequency (IDF): IDF is the measure of how important a term is. We need the IDF value because TF alone is not sufficient to understand the importance of words.

$idf_i$ = $log\frac{Number\ of\ Documents}{Number\ of\ documents\ with\ term\ 't'}$

Example:

Let's calculate the IDF value of word `this` in sentence 2.

IDF `this` in sentence 2 = $log\frac{Number\ of\ documents}{Number\ of\ documents\ containing\ the\ word\ this}$ = $log\frac{3}{3}$ = log(1) = 0

The IDF Values for the entire vocabulary would be:

| Term | Sentence 1 | Sentence 2 | Sentence 3 | IDF |
| ------ | ------ | -------- | ------ | ------ |
| This | 1 | 1 | 1 | $log\frac{3}{3}$ = 0 | 
| movie | 1 | 1 | 1 | $log\frac{3}{3}$ = 0 |
| is | 1 | 2 | 1 | $log\frac{3}{3}$, $log\frac{3}{3}$  = 0 |
| very | 1 | 0 | 0 | $log\frac{3}{1}$ = 0.48 |
| scary | 1 | 1 | 0 | $log\frac{3}{2}$ = 0.18 |
| and | 1 | 1 | 1 | $log\frac{3}{3}$ = 0|
| long | 1 | 0 | 0 | $log\frac{3}{1}$ = 0.48|
| not | 0 | 1 | 0 | $log\frac{3}{1}$ = 0.48|
| slow | 0 | 1 | 0 | $log\frac{3}{1}$ = 0.48|
| spooky | 0 | 0 | 1 | $log\frac{3}{1}$ = 0.48|
| good | 0 | 0 | 1 | $log\frac{3}{1}$ = 0.48|


We can observe that certain words such as "is", "the", and "and" have been assigned a value of 0, indicating their lower significance. In contrast, words such as "scary", "long", and "good" have a higher value, indicating their importance. By calculating the TF-IDF score for each word in the corpus, we can determine their respective importance levels.

$TF-IDF_{t,d}$ = $TF_{t,d}$  * $IDF_t$

| Term | Sentence 1 | Sentence 2 | Sentence 3 | IDF | TF-IDF Sentence 01 | TF-IDF Sentence 02 | TF-IDF Sentence 03 | 
| ------ | ------ | -------- | ------ | ------ | ------ | ------ | ------ | 
| This | 1 | 1 | 1 |  0 | 0 | 0 | 0 |
| movie | 1 | 1 | 1 |  0 | 0 | 0 | 0 |
| is | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 
| very | 1 | 0 | 0 | 0.48 | 0.068 | 0 | 0 | 
| scary | 1 | 1 | 0 | 0.18 | 0.025 | 0.022 | 0 |
| and | 1 | 1 | 1 | 0| 0 | 0 | 0 |
| long | 1 | 0 | 0 | 0.48| 0.068 | 0 | 0 
| not | 0 | 1 | 0 | 0.48| 0 | 0.060 | 0 |
| slow | 0 | 1 | 0 | 0.48| 0 | 0.060 | 0 |
| spooky | 0 | 0 | 1 | 0.48| 0 | 0 | 0.080|
| good | 0 | 0 | 1 | 0.48| 0 | 0 | 0.80 | 

After calculating the TF-IDF scores for our vocabulary, it became evident that less frequent words were given higher values, indicating their relative importance in the corpus. TF-IDF scores were found to be particularly high for words that were rare in all documents combined, but frequent in a single document, indicating their potential significance in that particular context.

#### 1.1.2. Problem of Bag of Worrds and TF-IDF
- Both BOW and TF-IDF approach semntic information is not stored. TF-IDF gives importance to uncommon words
- There is definately chance of overfitting 

To overcome such problem of BOW and TF-IDF we use technique called Word2Vec.

Word2vec

Word2Vec is a technique for natural language processing published in 2013 by Google. The word2vec algorithm uses a nueral network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a particular sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector.

- In the specific model, each word is basically represented as a vector of 32 or more dimension instead of a single word
- Here the semantic information and relation between different words is also preserved 


Let's discuss the differences between Bag of Words (BOW), Term Frequency-Inverse Document Frequency (TF-IDF), and Word2Vec. In BOW, we obtain a sparse matrix with either 0 or 1 values, while in TF-IDF, we may get decimal values ranging from 0 to 1. However, Word2Vec works differently. To illustrate this, let's consider a vocabulary comprising the unique words in a given corpus.

Vocabulary -> Unique Words -> Corpus 

Let's say vocabulary word I specifically have like something like BOY GIRL KING QUEEN APPLE MANGO
