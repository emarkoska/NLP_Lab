import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd


def read_corpus():
    sentences = np.concatenate((pd.read_table('data/multinli_dev_set.txt')['sentence1'].get_values(),
                                pd.read_table('data/multinli_dev_set.txt')['sentence2'].get_values()))

    labels = np.concatenate((pd.read_table('data/multinli_dev_set.txt')['genre'].get_values(),
                             pd.read_table('data/multinli_dev_set.txt')['genre'].get_values()))

    return sentences, labels


def plot_frequency():
    sentences, labels = read_corpus()
    flat_words_nonunique = []
    for sentence in sentences:
        flat_words_nonunique.append(nltk.word_tokenize(sentence.lower()))
    flat_words_nonunique = np.concatenate(flat_words_nonunique).ravel()
    flat_words = np.unique(flat_words_nonunique)
    print(flat_words.shape)

    frequency = nltk.probability.FreqDist(flat_words_nonunique)

    top = list(frequency)[:20]
    plt.plot(top)
    plt.show()


# ctrl+alt+l = indentation, formatting code

def main():
    plot_frequency()


if __name__ == '__main__':
    main()
