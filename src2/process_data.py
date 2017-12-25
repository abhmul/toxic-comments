import os
import re
import numpy as np
import pandas as pd
import argparse

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer

LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# Regex to remove all Non-Alpha Numeric and space
SPECIAL_CHARS = re.compile(r'[^a-z\d ]', re.IGNORECASE)
# regex to replace all numerics
NUMBERS = re.compile(r'\d+', re.IGNORECASE)


def load_data(path_to_data="../input"):
    path_to_train_csv = os.path.join(path_to_data, "train.csv")
    print("Reading in train data from %s" % path_to_train_csv)
    # Open the csv file
    train_csv = pd.read_csv(path_to_train_csv)
    train_ids = train_csv["id"].values
    train_texts = train_csv["comment_text"].fillna("NA").values
    train_labels = train_csv[LABEL_NAMES].values

    path_to_test_csv = os.path.join(path_to_data, "test.csv")
    print("Reading test data from %s" % path_to_test_csv)
    # Open the csv file
    test_csv = pd.read_csv(path_to_test_csv)
    test_ids = test_csv["id"].values
    test_texts = test_csv["comment_text"].fillna("NA").values

    return (train_ids, train_labels, train_texts), (test_ids, test_texts)


def parse_sentences(texts, sent_detector):
    # Parse out the sentences
    texts = [sent_detector(text) for text in texts]
    # Get the start sentences of each text
    text_starts = [i == 0 for text in texts for i, sent in enumerate(text)]
    # Flatten the texts
    flat_texts = [sent for text in texts for sent in text]
    return flat_texts, text_starts


def reformat_texts(flat_texts, text_starts):
    assert len(flat_texts) == len(text_starts)
    texts = []
    cur_text = []
    for i, sent in enumerate(flat_texts):
        cur_text.append(sent)
        # If this is our last text or marks the end of a text
        if i == (len(flat_texts) - 1) or text_starts[i+1]:
            # Add the current text and reset it
            texts.append(cur_text)
            cur_text = []
    return texts


def clean_text(text, remove_stopwords=False, remove_special_chars=True, replace_numbers=True, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case
    text = text.lower()
    # Optionally, remove stop words
    if remove_stopwords:
        text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]
        text = " ".join(text)

    # Remove Special Characters
    if remove_special_chars:
        text = SPECIAL_CHARS.sub('', text)

    # Replace Numbers
    if replace_numbers:
        text = NUMBERS.sub('n', text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text


def clean(texts, clean_func):
    return [clean_func(text) for text in texts]


def tokenize(train_texts, test_texts, tokenize_func=None, max_nb_words=100000):
    tokenizer = Tokenizer(num_words=max_nb_words)
    # Split up each text
    if tokenize_func is None:
        tokenize_func = lambda text: text.split()
    train_texts = [tokenize_func(text) for text in train_texts]
    test_texts = [tokenize_func(text) for text in test_texts]

    # Fit the tokenizer
    tokenizer.fit_on_sequences(train_texts + test_texts)

    # Tokenize the texts
    train_texts = [[tokenizer.word_index[word] for word in text] for text in train_texts]
    test_texts = [[tokenizer.word_index[word] for word in text] for text in test_texts]

    return train_texts, test_texts, tokenizer.word_index