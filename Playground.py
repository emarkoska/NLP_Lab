import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def clean_text(text):
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text


def vectorized_todf():
    training_texts = [
        "This is a good cat",
        "This is a bad day"
    ]

    test_texts = [
        "This day is a good day"
    ]

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text
    )

    # fit the vectorizer on the training text
    vectorizer.fit(training_texts)

    # get the vectorizer's vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

    # vectorization example
    df = pd.DataFrame(
        data=vectorizer.transform(test_texts).toarray(),
        index=["test sentence"],
        columns=vocabulary
    )

    print(df)

if __name__ == '__main__':
    vectorized_todf()
