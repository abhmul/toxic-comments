import os
import glob
import re
import numpy as np
import pandas as pd
import argparse
import pickle as pkl
from functools import partial
from tqdm import tqdm
import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from file_utils import safe_open_dir

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Clean and tokenize the data.')
parser.add_argument('-d', '--data', default="../input/", help='Path to the Toxic data')
parser.add_argument('-s', '--save', default="../processed_input/", help='Path to save the new data to.')
parser.add_argument('--parse_sent', action='store_true', help='Stores the data with sentences parsed out.')
parser.add_argument('--remove_stopwords', action='store_true', help='Removes the stop words from the data')
parser.add_argument('--keep_special_chars', action='store_true', help='Keeps special characters in the data')
parser.add_argument('--replace_numbers', action='store_true', help='Removes number digits in the data')
parser.add_argument('--stem_words', action='store_true', help='Keeps number digits in the data')
parser.add_argument('--max_nb_words', type=int, default=100000, help='Maximum number of words to keep in the data. ' +
                                                                     'Set to -1 to keep all words')
parser.add_argument('--nltk_tokenize', action='store_true', help="Uses the nltk punkt word tokenizer.")
parser.add_argument('--use_sklearn', action='store_true', help="Uses sklearn tokenizer and doesn't clean.")
parser.add_argument('--use_augmented', action='store_true', help="Processes additional augmented datasets")

LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# Regex to remove all Non-Alpha Numeric and space
SPECIAL_CHARS = re.compile(r'([^a-z\d ])', re.IGNORECASE)
# regex to replace all numerics
NUMBERS = re.compile(r'\d+', re.IGNORECASE)


def load_data(path_to_data="../input", train_name="train.csv", test_name="test.csv"):
    path_to_train_csv = os.path.join(path_to_data, train_name)
    # Open the csv file
    train_csv = pd.read_csv(path_to_train_csv)
    train_ids = train_csv["id"].values
    train_texts = train_csv["comment_text"].fillna("NA").values
    train_labels = train_csv[LABEL_NAMES].values
    logging.info("Loaded %s samples from %s" % (len(train_texts), path_to_train_csv))

    path_to_test_csv = os.path.join(path_to_data, test_name)
    logging.info("Reading test data from %s" % path_to_test_csv)
    # Open the csv file
    test_csv = pd.read_csv(path_to_test_csv)
    test_ids = test_csv["id"].values
    test_texts = test_csv["comment_text"].fillna("NA").values
    logging.info("Loaded %s samples from %s" % (len(test_texts), path_to_test_csv))

    return (train_ids, train_texts, train_labels), (test_ids, test_texts)


def load_text(text_name, path_to_data="../input"):
    path = os.path.join(path_to_data, text_name)
    csv = pd.read_csv(path)
    text = csv["comment_text"].fillna("NA").values
    logging.info("Loaded %s samples from %s" % (len(text), path))
    return text


def clean_text(text, remove_stopwords=False, substitute_special_chars=True, remove_special_chars=False,
               replace_numbers=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case
    text = text.lower()
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]
        text = " ".join(text)

    # Substitute special chars
    if substitute_special_chars:
        text = SPECIAL_CHARS.sub(r' \1 ', text)

    # Remove Special Characters
    if remove_special_chars and not substitute_special_chars:
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
    return [clean_func(text) for text in tqdm(texts)]


def tokenize(texts, tokenize_func=None, max_nb_words=-1):
    if max_nb_words == -1:
        max_nb_words = None
    tokenizer = Tokenizer(num_words=max_nb_words)
    # Split up each text
    # TODO For now using a tokenizer func is not suppoted
    if tokenize_func is not None:
        raise NotImplementedError("No tokenizer func is supported yet")

    # Fit the tokenizer
    logging.info("Fitting tokenizer...")
    tokenizer.fit_on_texts(texts)
    logging.info("Found %s words in texts" % len(tokenizer.word_index))
    return tokenizer.word_index


def process_texts(train_name, test_name, *augmentation_names, path_to_data="../input/", save_dest="../processed_input/",
                  remove_stopwords=False, substitute_special_chars=True, remove_special_chars=False,
                  replace_numbers=False, stem_words=False, tokenize_func=None,
                  max_nb_words=-1):

    # Initialize the cleaner
    logging.info("Loading and Cleaning the texts...")
    clean_func = partial(clean_text, remove_stopwords=remove_stopwords,
                         substitute_special_chars=substitute_special_chars,
                         remove_special_chars=remove_special_chars,
                         replace_numbers=replace_numbers, stem_words=stem_words)
    logging.info("Initializing a cleaner with settings: \n%r" % clean_func)

    # Load the texts
    augmentation_names = list(augmentation_names)
    text_names = [train_name, test_name] + augmentation_names
    texts = sum((clean(load_text(text_name, path_to_data=path_to_data), clean_func) for text_name in text_names), [])

    # Tokenize the texts
    logging.info("Tokenizing the texts...")
    word_index = tokenize(texts, tokenize_func=tokenize_func, max_nb_words=max_nb_words)
    # Cleanup the loaded texts
    logging.info("Cleaning up the loaded texts")
    del texts

    # Loading train and test data
    (train_ids, train_texts, train_labels), (test_ids, test_texts) = load_data(path_to_data=path_to_data,
                                                                               train_name=train_name,
                                                                               test_name=test_name)
    # Now reload the texts 1-by-1 to map them to their respective indicies
    logging.info("Mapping words to their respective indicies for augmentation texts...")
    for i, aug_name in enumerate(augmentation_names):
        text = [text_to_word_sequence(comment) for comment in
                clean(load_text(aug_name, path_to_data=path_to_data), clean_func)]
        text = [[word_index[word] for word in comment] for comment in tqdm(text)]
        raw_name = os.path.splitext(aug_name)[0]
        save_text(raw_name, train_ids, text, labels=train_labels, save_dest=save_dest)
        logging.info("Saved Augmentation Dataset %s/%s" % (i, len(augmentation_names)))
        # And some cleanup
        del text
    # Save the train and test texts now
    logging.info("Mapping and Saving train and test data...")
    train_texts = [text_to_word_sequence(comment) for comment in clean(train_texts, clean_func)]
    train_texts = [[word_index[word] for word in comment] for comment in tqdm(train_texts)]
    save_text("train", train_ids, train_texts, labels=train_labels, save_dest=save_dest)
    del train_texts
    del train_ids
    del train_labels
    test_texts = [text_to_word_sequence(comment) for comment in clean(test_texts, clean_func)]
    test_texts = [[word_index[word] for word in comment] for comment in tqdm(test_texts)]
    save_text("test", test_ids, test_texts, save_dest=save_dest)
    del test_ids
    del test_texts

    # Save the word index
    save_word_index(word_index, save_dest=save_dest)


def save_text(name, ids, text, labels=None, save_dest="../processed_input/"):
    safe_open_dir(save_dest)
    save_path = os.path.join(save_dest, name + ".npz")
    if labels is not None:
        np.savez(save_path, ids=ids, texts=text, labels=labels)
    else:
        np.savez(save_path, ids=ids, texts=text)
    logging.info("Saved %s" % name)


def save_word_index(word_index, save_dest="../processed_input/"):
    with open(os.path.join(save_dest, "word_index.pkl"), 'wb') as word_index_file:
        pkl.dump(word_index, word_index_file)
    logging.info("Saved Word Index")


if __name__ == "__main__":
    args = parser.parse_args()
    tokenize_function = nltk.tokenize.word_tokenize if args.nltk_tokenize else None
    train_augmentation_names = glob.glob(os.path.join(args.data, "train_*.csv")) if args.use_augmented else []
    train_augmentation_names = [os.path.basename(aug_name) for aug_name in train_augmentation_names]
    process_texts("train.csv", "test.csv", *train_augmentation_names, path_to_data=args.data, save_dest=args.save,
                  remove_stopwords=args.remove_stopwords, substitute_special_chars=(not args.keep_special_chars),
                  replace_numbers=args.replace_numbers, stem_words=args.stem_words,
                  tokenize_func=tokenize_function, max_nb_words=args.max_nb_words)
