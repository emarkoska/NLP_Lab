import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


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
    training_texts = ["This is a good cat", "This is a bad day"]
    test_texts = ["This day is a good day"]

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(stop_words="english", preprocessor=clean_text)

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


def load_train_test_imdb_data(data_dir):
    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in file_names:
                with open(os.path.join(path, f_name), "r", encoding="utf8") as f:
                    review = f.read()
                    data[split].append([review, score])

    np.random.shuffle(data["train"])
    data["train"] = pd.DataFrame(data["train"], columns=['text', 'sentiment'])

    np.random.shuffle(data["test"])
    data["test"] = pd.DataFrame(data["test"], columns=['text', 'sentiment'])

    return data["train"], data["test"]


def predict(train_data, test_data):
    # Transform each text into a vector of word counts
    vectorizer = CountVectorizer(stop_words="english", preprocessor=clean_text)

    training_features = vectorizer.fit_transform(train_data["text"])
    test_features = vectorizer.transform(test_data["text"])

    # Training
    model = LinearSVC()
    model.fit(training_features, train_data["sentiment"])
    y_pred = model.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_data["sentiment"], y_pred)

    print("Accuracy on the IMDB dataset: {:.2f}".format(acc * 100))


if __name__ == '__main__':
    train_data, test_data = load_train_test_imdb_data(data_dir="data/aclImdb/")
    predict(train_data, test_data)
