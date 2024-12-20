import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer


def read_corpus():
    sentences = np.concatenate((pd.read_table('data/multinli_dev_set.txt')['sentence1'].get_values(),
                                pd.read_table('data/multinli_dev_set.txt')['sentence2'].get_values()))

    labels = np.concatenate((pd.read_table('data/multinli_dev_set.txt')['genre'].get_values(),
                             pd.read_table('data/multinli_dev_set.txt')['genre'].get_values()))

    return sentences, labels

def plot_frequency_nonunique():
    sentences, labels = read_corpus()
    flat_words_nonunique = []
    for sentence in sentences:
        # This tokenizer allows for punctuation to remain
        flat_words_nonunique.append(nltk.word_tokenize(sentence.lower()))
    flat_words_nonunique = np.concatenate(flat_words_nonunique).ravel()
    flat_words = np.unique(flat_words_nonunique)
    plot_top20mostfrequent(flat_words_nonunique)

def plot_freq_nostopwordspunctuation():
    tokenizer = RegexpTokenizer(r'\w+')
    sentences, labels = read_corpus()
    flat_words_nonunique = []
    for sentence in sentences:
        # This tokenizer allows for punctiation to remain
        flat_words_nonunique.append(tokenizer.tokenize(sentence.lower()))
    flat_words_nonunique = np.concatenate(flat_words_nonunique).ravel()
    flat_words_noww = [x for x in flat_words_nonunique if
                       x not in stopwords.words('english')]  # Removing the stopwords from the word list
    plot_top20mostfrequent(flat_words_noww)

# ctrl+alt+l = indentation, formatting code

# Function removes the k least frequent words
def remove_k_words_andplot(k):
    # Should be refactored
    tokenizer = RegexpTokenizer(r'\w+')
    sentences, labels = read_corpus()
    flat_words_nonunique = []
    for sentence in sentences:
        # This tokenizer allows for punctiation to remain
        flat_words_nonunique.append(tokenizer.tokenize(sentence.lower()))
    flat_words_nonunique = np.concatenate(flat_words_nonunique).ravel()
    flat_words_noww = [x for x in flat_words_nonunique if
                       x not in stopwords.words('english')]  # Removing the stopwords from the word list

    flat_words_noww = [x for x in flat_words_nonunique if x not in stopwords.words('english')]
    freq = nltk.probability.FreqDist(flat_words_noww)
    list_minus_k = {}
    for i in freq:
        if freq[i] > k:
            list_minus_k[i] = freq[i]

    sorted_x = sorted(list_minus_k.items(), key=lambda kv: kv[1], reverse=True)

    x, y = zip(*sorted_x)
    plt.plot(x, y)


# Finds the stemming word for every word in corpus and plots the 20 most frequent ones.
def plot_mostfrequentstemmed():
    tokenizer = RegexpTokenizer(r'\w+')
    sentences, labels = read_corpus()
    flat_words_nonunique = []
    for sentence in sentences:
        # This tokenizer allows for punctiation to remain
        flat_words_nonunique.append(tokenizer.tokenize(sentence.lower()))
    flat_words_nonunique = np.concatenate(flat_words_nonunique).ravel()
    flat_words_noww = [x for x in flat_words_nonunique if x not in stopwords.words('english')]
    stemmed_words = []
    stemmer = PorterStemmer()
    for word in flat_words_noww:
        stemmed_words.append(stemmer.stem(word))

    stemmed_words = list(stemmed_words)
    plot_top20mostfrequent(stemmed_words)

def plot_top20mostfrequent(dict):
    frequency = nltk.probability.FreqDist(dict)  # Returns tuples of word : frequency
    first20pairs = {k: frequency[k] for k in list(frequency)[:20]}  # Returns list of the first 20 pairs
    sorted_x = sorted(first20pairs.items(), key=lambda kv: kv[1],
                      reverse=True)  # Sorted in reverse so we have the most frequent words
    x, y = zip(*sorted_x)  # Separating in two lists
    plt.plot(x, y)
    plt.show()


def main():
    plot_mostfrequentstemmed()




if __name__ == '__main__':
    main()
