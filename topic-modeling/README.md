# Topic Modeling [[view code]](http://nbviewer.jupyter.org/github/marcotav/unsupervised-learning/blob/master/topic-modeling/notebooks/topic-modeling-lda.ipynb) 
![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/ntlk-v3.2.5-yellow.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg) ![Image title](https://img.shields.io/badge/gensim-0.3.4-blue.svg)

**The code is available [here](http://nbviewer.jupyter.org/github/marcotav/unsupervised-learning/blob/master/topic-modeling/notebooks/topic-modeling-lda.ipynb) or by clicking on the [view code] link above.**

<p align="center">
  <a href="#intro"> Introduction </a> •
  <a href="#lib"> Libraries </a> •
  <a href="#pro"> Problem domain </a> •
  <a href="#cle"> Cleaning the text </a> •
  <a href="#docmatrix"> Document-term matrix </a> •
  <a href="#model"> LDA model </a>  •
  <a href="#pyLDAvis"> pyLDAvis </a>  •
  <a href="#results"> Results </a> 
</p>



<a id = 'intro'></a>
## Introduction

In this notebook, I will use Python and its libraries for **topic modeling**, an unsupervised approach which can be used to mine unstructured data, fetching the information we need. More concretely, in topic modeling, statistical models are used to identify topics or categories in a document or a set of documents, finding hidden patterns in a *[text corpus](https://en.wikipedia.org/wiki/Text_corpus)* (a text corpus is just "a large and structured set of texts"). It is useful for document clustering, organizing text data and to retrieve information from unstructured text.

Topics are "a repeating pattern of co-occurring terms in a corpus". An example of good topic model would attribute say, "gravity", "Newton" and "electron" to a topic, in this case Physics. 

I will use one specific method called **Latent Dirichlet Allocation (LDA)** which is a matrix factorization technique. Any corpus can be represented by a matrix of document terms such as (the transpose) of [this one below](http://www.speedlab.io/en/2017/03/08/creation-semantic-variables-based-document-term-matrix-dtm/) . This matrix shows a corpus of 5 documents with vocabulary size of 5 words. LDA converts this matrix into two lower dimensional matrices, a document-topics matrix and a topic-words matrix

<br/>
<p align="center">
  <img src='images/DTM.png' width="400">
</p>

The algorithm can be summarized as follows:
- First we select - without previous knowledge about what the topics actually are - a fixed number of topics T 
- We then randomly assign each word to some topic
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

I will now generate an LDA model and for that, the frequency that each term occurs within each document needs to be understood. A **document-term matrix** is constructed to do that. It contains a corpus of n documents and a vocabulary of m words. Each cell ij counts the frequency of the word j in the document i. Converting tokens into a dictionary:

```
from gensim import corpora, models
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(text) for text in tokens]
```
The function `Dictionary( )` traverses `tokens` and assigns an integer id to each on, while collecting word counts. The function `doc2bow` coverts the dictionary into a bag-of-words resulting in a list of vectors, one for each document. In each vector there is a set of tuples. The tuples are of the form (term ID, term frequency).

Examining our corpus:
```
corpus[0]
corpus[1]
```
gives:
```
[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
[(6, 1), (7, 1), (8, 1), (9, 1)]
```

<a id = 'model'></a>
## LDA model

The parameters are:
- num_topics: how many topics should be generated. 
- id2word: previous dictionary to map ids to strings.
- passes: Laps the model will take through the corpus More passes, more accuracy.

```
import gensim
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
for el in ldamodel.print_topics(num_topics=10, num_words=5):
    print(el,'\n')
```
The output is:

```
(0, '0.092*"network" + 0.063*"wireless" + 0.030*"sensor" + 0.018*"routing" + 0.018*"mobile"') 

(1, '0.028*"power" + 0.023*"design" + 0.017*"based" + 0.016*"network" + 0.014*"using"') 

(2, '0.029*"based" + 0.027*"video" + 0.024*"coding" + 0.020*"algorithm" + 0.018*"image"') 

(3, '0.027*"algorithm" + 0.019*"system" + 0.014*"based" + 0.014*"design" + 0.013*"environment"') 

(4, '0.021*"service" + 0.020*"network" + 0.019*"content" + 0.019*"social" + 0.017*"aware"') 

(5, '0.025*"system" + 0.020*"model" + 0.016*"based" + 0.015*"animation" + 0.011*"access"') 

(6, '0.020*"application" + 0.017*"based" + 0.016*"system" + 0.015*"delta" + 0.015*"sigma"') 

(7, '0.025*"using" + 0.020*"search" + 0.017*"structure" + 0.016*"database" + 0.016*"engine"') 

(8, '0.029*"large" + 0.028*"system" + 0.022*"database" + 0.016*"scale" + 0.014*"management"') 

(9, '0.028*"query" + 0.024*"approach" + 0.022*"system" + 0.021*"based" + 0.020*"using"')  
```

<a id = 'pyLDAvis'></a>
## pyLDAvis
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
  <img src='images/pyLDAvis_new.png' width="900">
</p>


<a id = 'results'></a>
## Results

Final analysis to be finished.
