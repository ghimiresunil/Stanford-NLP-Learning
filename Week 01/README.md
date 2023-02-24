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

- Tokenization: First, the input text is tokenized. A sentence is represented as a list of its constituent words, and it’s done for all the input sentences.

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

Text processing is the initial step, but our main focus is on how to derive vectors using bag of words. 🤔

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

Finally we have derived vectors using bag of words.😊 While bag of words is a useful technique for text analysis, it also has its disadvantages.  One of the main disadvantages is that it doesn't take into account the context in which words appear. As a result, words with multiple meanings can be assigned the same vector representation, leading to ambiguity in the analysis. Additionally, bag of words doesn't capture the relationship between words, such as synonyms and antonyms, which can affect the accuracy of natural language processing tasks.

<b> Listed drawback of using a bag of words </b>

- If the new sentences contain new words, then our vocabulary size would increase and thereby, the length of vector would increase too.
- Additionally, the vectors would also contain many 0s, thereby resulting in a sparse matrix (which is what we would like to avoid)
- We are retaining no information on the grammar of the sentences nor the ordering of the words in the text.

From the above-generated vectors, it can be observed that the values assigned to the vectors are either 1 or 0. However, an important point to note is that both "intelligent" and "boy" have been assigned a value of 1, despite having different semantic meanings. This makes it difficult to determine which word holds greater importance. In sentiment analysis, it is crucial to identify the words that carry more weightage in determining the sentiment.

To overcome some of the limitations of the bag-of-words model we have something call as `TF-IDF` which is also called `Term Frequency and inverse document frequency` 😊 

Term Frequency - Inverse Document Frequency (TF-IDF) is a numerical statistics that is intended to reflect how important a word is to a document in a collection or corpus. 

Term Frequency (TF) - If is a measure of how frequently a term, t appears in a document d. 

$$t_{f_{t_i}d} = \frac{n_{t_id}}{Number\ of\ terms\ in\ the\ document}$$

Here, in the numerator, n is the number of times the term 't' appears in the document 'd'. Thus, each document and term would have its own TF value.
