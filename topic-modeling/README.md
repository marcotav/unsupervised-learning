# [Topic Modeling](http://nbviewer.jupyter.org/github/marcotav/unsupervised-learning/blob/master/topic-modeling/notebooks/topic-modeling-lda.ipynb) ![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/ntlk-v3.2.5-yellow.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg) ![Image title](https://img.shields.io/badge/gensim-0.3.4-blue.svg)

**For the best viewing experience use [nbviewer]().**


<p align="center">
  <a href="#intro"> Introduction </a> •
  <a href="#lib"> Libraries </a> •
  <a href="#pro"> Problem domain </a> •
  <a href="#cle"> Cleaning the text </a> •
  <a href="#docmatrix"> Document-term matrix </a> •
  <a href="#model"> LDA model </a> 
</p>

<a id = 'intro'></a>
## Introduction

In this notebook, I will use Python and its libraries for **topic modeling**. In topic modeling, statistical models are used to identify topics or categories in a document or a set of documents. I will use one specific method called **Latent Dirichlet Allocation (LDA)**. 

The algorithm can be summarized as follows:
- First we select - without previous knowledge regarding what the topics actually are - a fixed number of topics T 
- We then randomly assign each word to a topic
- For each document d, word w and topic t we calculate the probability P(t|w,d) that the word w of document d corresponds to topic t
- We then reassign each word w to some topic based on  P(t|w,d) and repeat the process until we find the optimal assignment of words to topics

<a id = 'lib'></a>
## Libraries  

This notebook uses the following packages:

- `spacy`
- `nltk`
- `random`
- `gensim`
- `pickle`
- `pandas`
- `sklearn`

<a id = 'pro'></a>
## Problem domain

In this project I apply LDA to labels on research papers. The dataset is a subset of [this](https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/dataset.csv) data set.

In this projects I will use the `spaCy` library. `spaCy` is:

> An industrial-strength natural language processing (NLP) library for Python. spaCy's goal is to take recent advancements in natural language processing out of research papers and put them in the hands of users to build production software.

```
import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()
```
<a id = 'imp'></a>
## Importing the documents

```
df = pd.read_csv('articles.csv',header=None)
```
From `df` I will build a list `doc_set` containing the row entries:

```
doc_set = df.values.T.tolist()[0]
```
<a id = 'cle'></a>
## Cleaning the text

Before applying natural language processing tools to our problem, I will provide a quick review of some basic procedures using Python. We first import `nltk` and the necessary classes for lemmatization and stemming:

```
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
```

We then create objects of the classes `PorterStemmer` and `WordNetLemmatizer`:

```
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
```

Tokenizing using Regex, building lists of tokens and lemmatizing:
```
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
tokenined_docs = []
for doc in doc_set:
    tokens = tokenizer.tokenize(doc.lower())
    tokenined_docs.append(tokens)
lemmatized_tokens = []
for lst in tokenined_docs:
    tokens_lemma = [lemmatizer.lemmatize(i) for i in lst]
    lemmatized_tokens.append(tokens_lemma)
```    

Dropping stopwords and words with less than n letters:

```from stop_words import get_stop_words
en_stop_words = get_stop_words('en')
n=4
tokens = []
for lst in lemmatized_tokens:
    tokens.append([i for i in lst if not i in en_stop_words if len(i) > n])
```

<a id = 'docmatrix'></a>
## Document-term matrix

I will now generate an LDA model and for that, the frequency that each term occurs within each document needs to be understood. A **document-term matrix** is constructed to do that. It contains a corpus of $n$ documents and a vocabulary of $m$ words. Each cell $ij$ counts the frequency of the word $j$ in the document $i$. Converting tokens into dictionary:

```
from gensim import corpora, models
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(text) for text in tokens]
```
<a id = 'model'></a>
## LDA model

```
import gensim
ldamodel_3 = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
for el in ldamodel_3.print_topics(num_topics=3, num_words=3):
    print(el,'\n')
```
The output is:

```
(0, '0.017*"system" + 0.017*"image" + 0.017*"based"') 

(1, '0.035*"network" + 0.015*"multi" + 0.012*"based"') 

(2, '0.016*"based" + 0.013*"using" + 0.012*"system"') 
```

Now I use `pyLDAvis`:
```
import pyLDAvis.gensim
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model3.gensim')
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
```


<br/>
<p align="center">
  <img src='https://github.com/marcotav/unsupervised-learning/blob/master/topic-modeling/images/puLDAvis.png' width="900">
</p>

## Final analysis to be finished
