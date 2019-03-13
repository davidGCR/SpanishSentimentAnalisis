
import numpy as np
from sklearn.decomposition import PCA
import gensim.models.word2vec
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import csv
import pandas as pd
import string, re
import pickle


## Functions for cleaning data
def remove_mentions(input_text):
    return re.sub(r'@\w+', '', input_text)
    
def remove_urls(input_text):
    return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

def remove_digits(input_text):
    return re.sub('\d+', '', input_text)
    
def stemming(input_text):
    porter = PorterStemmer()
    words = input_text.split() 
    
    stemmed_words = [porter.stem(word) for word in words]
    return " ".join(stemmed_words)
    
def remove_stopwords(input_text):
    stopwords_list = stopwords.words('spanish')
    words = input_text.split() 
    whitelist = ["no"]
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 

def replace_accents(input_text):
    
    input_text = re.sub(u"[àáâãäå]", 'a', input_text)
    input_text = re.sub(u"[èéêë]", 'e', input_text)
    input_text = re.sub(u"[ìíîï]", 'i', input_text)
    input_text = re.sub(u"[òóôõö]", 'o', input_text)
    input_text = re.sub(u"[ùúûü]", 'u', input_text)
    input_text = re.sub(u"[ýÿ]", 'y', input_text)
    return input_text

def to_lower(input_text):
    return input_text.lower()

def remove_punctuation(input_text):
    # Make translation table
    punct = string.punctuation
    punct = punct+'¡'+'?'+'¿'+'…'
    trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
    input_text = re.sub("[\.][\.][\.]", " ", input_text)
    return input_text.translate(trantab)


def processJaja(word):
    dirtyJaja = re.compile(r'[ja]{5,}')
    dirtyJeje = re.compile(r'[je]{5,}')
    dirtyHihi = re.compile(r'[hi]{5,}')
    dirtyJaj = re.compile(r'[a-zA-z]+ja{2,}')
    while dirtyJaja.search(word)!=None:
        word = word.replace(dirtyJaja.search(word).group(),'jaja')
    while dirtyJeje.search(word)!=None:
        word = word.replace(dirtyJeje.search(word).group(),'jaja')
    while dirtyHihi.search(word)!=None:
        word = word.replace(dirtyHihi.search(word).group(),'jaja')
    while dirtyJaj.search(word)!=None:
        word = word.replace(dirtyJaj.search(word).group(),'jaja')
    
    return word    

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

#aplanar
def flattern(A):
    
    'Flattens a list of lists and strings into a list.'
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flattern(i))
        else: rt.append(i)
    return rt


# Remove for Hashtags
def pefix_hash_tag(tweets):
    tweets = [[['hash_tag', i] if i.startswith('#') else i for i in tweet.split()] for tweet in tweets]
    tweets = np.array([flattern(tweet) for tweet in tweets])
    tweets = np.array([' '.join(i) for i in tweets])
    return tweets

## Cleaning data
def clean_data(list_sentences):
    clean_tweet_texts = []
    i=0
    for sentence  in list_sentences:
        print('**- ',i,' ',sentence)
        if sentence is not None:
            clean_text = remove_mentions(sentence)
            clean_text = remove_urls(clean_text)
            clean_text = remove_punctuation(clean_text)
            clean_text = replace_accents(clean_text)
            clean_text = remove_digits(clean_text)
            clean_text = to_lower(clean_text)
            clean_text = remove_stopwords(clean_text)
            clean_text = processJaja(clean_text)
            clean_text = remove_emoji(clean_text)
            #clean_text = stemming(clean_text)
            clean_tweet_texts.append(clean_text)
        i+=1
    return clean_tweet_texts

def clean_sentence(sentence):
    if sentence is not None:
        clean_text = remove_mentions(sentence)
        clean_text = remove_urls(clean_text)
        clean_text = remove_punctuation(clean_text)
        clean_text = replace_accents(clean_text)
        clean_text = remove_digits(clean_text)
        clean_text = to_lower(clean_text)
        clean_text = remove_stopwords(clean_text)
        clean_text = processJaja(clean_text)
        clean_text = remove_emoji(clean_text)
    return clean_text

def char_count(word, chars, lbound=2):
    char_count = [word.count(char) for char in chars]
    return all(i >= lbound for i in char_count)

def replace_lol(repl_str, texts):
    for string, chars in repl_str:
        texts = [[[string, i] if char_count(i, set(chars), 2) else i for i in text.split()] for text in texts]
        texts = np.array([flattern(text) for text in texts])
        texts = np.array([' '.join(text) for text in texts])
    return texts

# Lol type characters
repl_str = [('risa_ja','ja'), ('risa_ji','ji'), ('risa_je','je'), ('risa_jo','jo'), ('risa_ju', 'ju')]

# Adding prefix to lol type characters
def add_prefix_lol(tweets):
    tweets = replace_lol(repl_str, tweets)
    return tweets