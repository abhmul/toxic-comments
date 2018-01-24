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
from sklearn.feature_extraction.text import CountVectorizer

from file_utils import safe_open_dir

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Clean and tokenize the data.')
parser.add_argument('-d', '--data', default="../input/", help='Path to the Toxic data')
parser.add_argument('-s', '--save', default="../processed_input/", help='Path to save the new data to.')
parser.add_argument('--parse_sent', action='store_true', help='Stores the data with sentences parsed out.')
parser.add_argument('--remove_stopwords', action='store_true', help='Removes the stop words from the data')
parser.add_argument('--keep_special_chars', action='store_true', help='Keeps special characters in the data')
parser.add_argument('--keep_numbers', action='store_true', help='Keeps number digits in the data')
parser.add_argument('--substitute_special_chars', action='store_true', help='Substitutes the special chars for \1')
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
    logging.info("Reading in train data from %s" % path_to_train_csv)
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


def load_augmentation_texts(path_to_data="../input", train_name="train_*.csv"):
    path_to_train_csv = os.path.join(path_to_data, train_name)
    augmentation_paths = glob.glob(path_to_train_csv)
    augmentation_texts = []
    for path in augmentation_paths:
        logging.info("Reading in augmentation data from %s" % path)
        aug_csv = pd.read_csv(path)
        augmentation_texts.append(aug_csv["comment_text"].fillna("NA").values)
        logging.info("Loaded %s samples from %s" % (len(augmentation_texts[-1]), path))
    return augmentation_texts


def parse_sentences(texts, sent_detector):
    # Parse out the sentences
    texts = [sent_detector.tokenize(text) for text in tqdm(texts)]
    # Get the start sentences of each text
    text_lens = [len(text) for text in texts]
    # Flatten the texts
    flat_texts = [sent for text in texts for sent in text]
    return flat_texts, text_lens


def reformat_texts(flat_texts, text_lens):
    assert len(flat_texts) == sum(text_lens)
    texts = []
    cur_text = []
    cur_len_ind = 0
    for sent in flat_texts:
        cur_text.append(sent)
        # If we have added the correct number of sentences
        if len(cur_text) == text_lens[cur_len_ind]:
            texts.append(cur_text)
            cur_text = []
            cur_len_ind += 1
    assert cur_text == []
    return texts


def clean_text(text, remove_stopwords=False, substitute_special_chars=False, remove_special_chars=True, replace_numbers=True, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case
    text = text.lower()
    # Optionally, remove stop words
    if remove_stopwords:
        text.split()
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


def tokenize(train_texts, test_texts, augmentation_texts=tuple(), tokenize_func=None, max_nb_words=100000):
    if max_nb_words == -1:
        max_nb_words = None
    tokenizer = Tokenizer(num_words=max_nb_words)
    # Split up each text
    # TODO For now using a tokenizer func is not suppoted
    if tokenize_func is not None:
        raise NotImplementedError("No tokenizer func is supported yet")
    # if tokenize_func is None:
    #     tokenize_func = lambda text: text.split()
    # train_texts = [tokenize_func(text) for text in train_texts]
    # test_texts = [tokenize_func(text) for text in test_texts]

    # Fit the tokenizer
    logging.info("Fitting tokenizer...")
    tokenizer.fit_on_texts(sum((train_texts, test_texts, *augmentation_texts), []))
    logging.info("Found %s words in texts" % len(tokenizer.word_index))

    # Tokenize the texts
    logging.info("Converting texts to word sequences...")
    train_texts = [text_to_word_sequence(text) for text in tqdm(train_texts)]
    test_texts = [text_to_word_sequence(text) for text in tqdm(test_texts)]
    # Tokenize the augmentation texts
    augmentation_texts = [[text_to_word_sequence(text) for text in tqdm(aug_text)] for aug_text in augmentation_texts]
    return train_texts, test_texts, tokenizer.word_index, augmentation_texts


def process_texts(train_texts, test_texts, augmentation_texts=tuple(), parse_sent=False, remove_stopwords=False,
                  substitute_special_chars=False,
                  remove_special_chars=True, replace_numbers=True, stem_words=False, tokenize_func=None,
                  max_nb_words=100000):
    # Parse out the sentences if we need to
    if parse_sent:
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        logging.info("Parsing sentences is activated")
        train_texts, train_lens = parse_sentences(train_texts, sent_detector)
        test_texts, test_lens = parse_sentences(test_texts, sent_detector)
        augmentation_lens = []
        new_augmentation_texts = []
        for aug_text in augmentation_texts:
            aug_text, aug_lens = parse_sentences(aug_text, sent_detector)
            augmentation_lens.append(aug_lens)
            new_augmentation_texts.append(aug_text)
        augmentation_texts = new_augmentation_texts

    # Clean the texts
    logging.info("Cleaning the texts...")
    clean_func = partial(clean_text, remove_stopwords=remove_stopwords, substitute_special_chars=substitute_special_chars,
                         remove_special_chars=remove_special_chars,
                         replace_numbers=replace_numbers, stem_words=stem_words)
    train_texts = clean(train_texts, clean_func)
    test_texts = clean(test_texts, clean_func)
    # Clean the augmentation texts
    augmentation_texts = tuple(clean(aug_text, clean_func) for aug_text in augmentation_texts)

    # Tokenize the texts
    logging.info("Tokenizing the texts...")
    output = tokenize(train_texts, test_texts, augmentation_texts=augmentation_texts,
                      tokenize_func=tokenize_func, max_nb_words=max_nb_words)
    train_texts, test_texts, word_index, augmentation_texts = output

    clean_train_texts, clean_test_texts = train_texts, test_texts
    if parse_sent:
        clean_train_texts = reformat_texts(clean_train_texts, train_lens)
        clean_test_texts = reformat_texts(clean_test_texts, test_lens)
        # Turn them into the original docs
        clean_train_texts = [[word for sent in text for word in sent] for text in clean_train_texts]
        clean_test_texts = [[word for sent in text for word in sent] for text in clean_test_texts]
    save_clean_text(clean_train_texts, clean_test_texts, save_dest=args.save)

    # Map the texts to their indicies
    logging.info("Mapping words to their respective indicies...")
    train_texts = [[word_index[word] for word in text] for text in tqdm(train_texts)]
    test_texts = [[word_index[word] for word in text] for text in tqdm(test_texts)]
    augmentation_texts = [[[word_index[word] for word in text] for text in tqdm(aug_text)] for aug_text in augmentation_texts]

    # Reformat the texts if necessary
    if parse_sent:
        logging.info("Reformatting flat texts to their sentence parsed forms")
        train_texts = reformat_texts(train_texts, train_lens)
        test_texts = reformat_texts(test_texts, test_lens)
        augmentation_texts = [reformat_texts(aug_text, aug_lens) for aug_text, aug_lens in
                              zip(augmentation_texts, augmentation_lens)]

    return train_texts, test_texts, word_index, augmentation_texts


def save_clean_text(train_texts, test_texts, save_dest="../processed_input/"):
    safe_open_dir(save_dest)
    np.save(os.path.join(save_dest, "clean_train.npy"), train_texts)
    np.save(os.path.join(save_dest, "clean_test.npy"), test_texts)


def save_data(train_data, test_data, word_index, augmentation_texts=tuple(), save_dest="../processed_input/"):
    train_ids, train_texts, train_labels = train_data
    test_ids, test_texts = test_data
    # Save the datasets
    safe_open_dir(save_dest)
    train_save_dest = os.path.join(save_dest, "train.npz")
    np.savez(train_save_dest, ids=train_ids, texts=train_texts, labels=train_labels)
    logging.info("Saved Train Dataset")
    # Save the augmentation
    for i, aug_text in enumerate(augmentation_texts, start=1):
        aug_save_dest = os.path.join(save_dest, "train_aug%s.npz" % i)
        np.savez(aug_save_dest, ids=train_ids, texts=aug_text, labels=train_labels)
        logging.info("Saved Augmentation Dataset %s/%s" % (i, len(augmentation_texts)))
    # Save the text data
    test_save_dest = os.path.join(save_dest, "test.npz")
    np.savez(test_save_dest, ids=test_ids, texts=test_texts)
    logging.info("Saved Test Dataset")
    # Save the word_index
    with open(os.path.join(save_dest, "word_index.pkl"), 'wb') as word_index_file:
        pkl.dump(word_index, word_index_file)
    logging.info("Saved Word Index")


if __name__ == "__main__":
    args = parser.parse_args()
    (train_ids, train_texts, train_labels), (test_ids, test_texts) = load_data(args.data)
    augmentation_texts = load_augmentation_texts(args.data) if args.use_augmented else tuple()
    tokenize_func = nltk.tokenize.word_tokenize if args.nltk_tokenize else None
    train_texts, test_texts, word_index, augmentation_texts = process_texts(train_texts, test_texts,
                                                                            augmentation_texts=augmentation_texts,
                                                                            parse_sent=args.parse_sent,
                                                                            remove_stopwords=args.remove_stopwords,
                                                                            substitute_special_chars=args.substitute_special_chars,
                                                                            remove_special_chars=(not args.keep_special_chars),
                                                                            replace_numbers=(not args.keep_numbers),
                                                                            stem_words=args.stem_words, tokenize_func=tokenize_func,
                                                                            max_nb_words=args.max_nb_words)
    train_data = (train_ids, train_texts, train_labels)
    test_data = (test_ids, test_texts)
    save_data(train_data, test_data, word_index, augmentation_texts=augmentation_texts, save_dest=args.save)