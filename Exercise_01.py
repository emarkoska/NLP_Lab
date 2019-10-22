import numpy as np
import pandas as pd


def read_corpus():
    sentences = np.concatenate((pd.read_table('data/multinli_dev_set.txt')['sentence1'].get_values(),
                                pd.read_table('data/multinli_dev_set.txt')['sentence2'].get_values()))

    labels = np.concatenate((pd.read_table('data/multinli_dev_set.txt')['genre'].get_values(),
                             pd.read_table('data/multinli_dev_set.txt')['genre'].get_values()))

    return sentences, labels


# ctrl+alt+l = indentation, formatting code

def main():
    sentences, labels = read_corpus()
    print(sentences)


if __name__ == '__main__':
    main()
