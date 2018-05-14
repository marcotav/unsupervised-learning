# [Topic Modeling](http://nbviewer.jupyter.org/github/marcotav/unsupervised-learning/blob/master/topic-modeling/notebooks/topic-modeling-lda.ipynb) ![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/ntlk-v3.2.5-yellow.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg) ![Image title](https://img.shields.io/badge/gensim-0.3.4-blue.svg)

**For the best viewing experience use [nbviewer]().**


<p align="center">
  <a href="#intro"> Introduction </a> •
  <a href="#chal"> Challenges and Applications </a> •
  <a href="#over"> Overview and Data </a> •
  <a href="#orgdata"> Preprocessing the data </a> •
  <a href="#Small datasets"> Problems with small datasets </a> •
  <a href="#bn"> Using bottleneck features of Inception V3 </a> •
  <a href="#InceptionV3"> Creating and training using the InceptionV3 model </a> •
  <a href="#trainfully"> Training the fully-connected network</a> •
  <a href="#plots"> Plotting the accuracy and loss histories </a> •
  <a href="#Conclusions">Conclusions</a> •
  <a href="#td">To Dos</a> 
</p>

<a id = 'intro'></a>
## Introduction

[[go back to the top]](#Table-of-contents)

In this notebook, I will use Python and its libraries for **topic modeling**. In topic modeling, statistical models are used to identify topics or categories in a document or a set of documents. I will use one specific method called **Latent Dirichlet Allocation (LDA)**. The algorithm can be summarized as follows:
- First we select - without previous knowledge regarding what the topics actually are - a fixed number of topics $T$ 
- We then randomly assign each word to a topic
- For each document $d$, word $w$ and topic $t$ we calculate the probability $P(t\,|\,w,d)$ that the word $w$ of document $d$ corresponds to topic $t$
- We then reassign each word $w$ to some topic based on $P(t\,|\,w,d)$ and repeat the process until we find the optimal assignment of words to topics

## Libraries  

[[go back to the top]](#Table-of-contents)

This notebook uses the following packages:

- `spacy`
- `nltk`
- `random`
- `gensim`
- `pickle`
- `pandas`
- `sklearn`

import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # see the value of multiple statements at once.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

## The problem domain

[[go back to the top]](#Table-of-contents)

In this project I apply LDA to labels on research papers. The dataset is a subset of [this](https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/dataset.csv) data set.

## Using `spaCy`

[[go back to the top]](#Table-of-contents)

In this projects I will use the `spaCy` library (see this [link](https://github.com/skipgram/modern-nlp-in python/blob/master/executable/Modern_NLP_in_Python.ipynb)). 

`spaCy` is:

> An industrial-strength natural language processing (NLP) library for Python. spaCy's goal is to take recent advancements in natural language processing out of research papers and put them in the hands of users to build production software.

import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()

## Importing the documents

[[go back to the top]](#Table-of-contents)

df = pd.read_csv('articles.csv',header=None)
df.columns = ['titles']
df.shape
df.head()

## List of documents

[[go back to the top]](#Table-of-contents)

From `df` I will build a list `doc_set` containing the row entries:

doc_set = df.values.T.tolist()[0]
print(doc_set[0:10])

## Cleaning the text

[[go back to the top]](#Table-of-contents)

Before applying natural language processing tools to our problem, I will provide a quick review of some basic procedures using Python. We first import `nltk` and the necessary classes for lemmatization and stemming:

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

We then create objects of the classes `PorterStemmer` and `WordNetLemmatizer`:

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

To use lemmatization and/or stemming in a given string text we must first tokenize it. The code below matches word characters until it reaches a non-word character, like a space. 

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

### Build a list of lists of tokens 

tokenined_docs = []
for doc in doc_set:
    tokens = tokenizer.tokenize(doc.lower())
    tokenined_docs.append(tokens)
    
print(tokenined_docs[0:3])

### Apply lemmatizing

lemmatized_tokens = []
for lst in tokenined_docs:
    tokens_lemma = [lemmatizer.lemmatize(i) for i in lst]
    lemmatized_tokens.append(tokens_lemma)
    
print(lemmatized_tokens[0:3])

### Dropping stopwords and words with less than $n$ letters

from stop_words import get_stop_words
en_stop_words = get_stop_words('en')

n=4
tokens = []
for lst in lemmatized_tokens:
    tokens.append([i for i in lst if not i in en_stop_words if len(i) > n])

print(tokens[0:3])

## Document-term matrix

[[go back to the top]](#Table-of-contents)

I will now generate an LDA model and for that, the frequency that each term occurs within each document needs to be understood.

A **document-term matrix** is constructed to do that. It contains a corpus of $n$ documents and a vocabulary of $m$ words. Each cell $ij$ counts the frequency of the word $j$ in the document $i$.

|               | word_1 | word_2 | ... | word_m |
| ------------- |:------:| ----- :|----- :|----- :|
| doc_1         | 1      | 3   | ... |2
| doc_2         | 2      |   3   |...|3
| ...           | ...    |    2   |...|1
| doc_n         | 1      |    1   |...|1

What LDA does is to convert this matrix into two matrices with lower dimensions namely:

|               | topic_1 | topic_2 | ... | topic_T |
| ------------- |:------:| ----- :|----- :|----- :|
| doc_1         | 0      | 1   | ... |1
| doc_2         | 0      |   1   |...|1
| ...           | ...    |    ...   |...|1
| doc_n         | 1      |    0   |...|0

and

|               | word_1 | word_2 | ... | word_m |
| ------------- |:------:| ----- :|----- :|----- :|
| topic_1         | 1      | 0   | ... |1
| topic_2         | 1      |   0   |...|1
| ...           | ...    |    ...   |...|1
| topic_T         | 1      |    1   |...|1




## Tokens into dictionary

[[go back to the top]](#Table-of-contents)

from gensim import corpora, models

dictionary = corpora.Dictionary(tokens)

## Tokenize documents into document-term matrix

[[go back to the top]](#Table-of-contents)

corpus = [dictionary.doc2bow(text) for text in tokens]

import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

corpus[0]

## LDA model

import gensim
ldamodel_3 = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
ldamodel_3.save('model3.gensim')
ldamodel_4 = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
ldamodel_4.save('model4.gensim')

for el in ldamodel_3.print_topics(num_topics=3, num_words=3):
    print(el,'\n')

for el in ldamodel_4.print_topics(num_topics=3, num_words=3):
    print(el,'\n')

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')

corpus = pickle.load(open('corpus.pkl', 'rb'))

lda = gensim.models.ldamodel.LdaModel.load('model3.gensim')

import pyLDAvis.gensim

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

pyLDAvis.display(lda_display)
