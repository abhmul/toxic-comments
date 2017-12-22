import os
import pickle as pkl
import argparse
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec

parser = argparse.ArgumentParser(description='Tokenize and Embed Data.')
parser.add_argument('-w', '--word2vec',
                    help='Path to word2vec embeddings')
parser.add_argument('-d', '--data',
                    help='Path to the Toxic data')
parser.add_argument('-s', '--save',
                    help='Path to save the new data to.')
parser.add_argument('-t', '--tokenize', action='store_true', default=False,
                    help='Stores the data with sentences parsed out.')
args = parser.parse_args()

LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def tokenize_texts(texts, sent_detector=None):
    for text in tqdm(texts):
        # Tokenize the text
        try:
            if sent_detector is not None:
                tokens = [nltk.tokenize.word_tokenize(sent) for sent in sent_detector.tokenize(text)]
            else:
                tokens = nltk.tokenize.word_tokenize(text)
        except:
            print(text)
            raise ValueError()
        yield tokens


def load_data(vocab, path_to_data="../input/", parse_sentences=False):
    encoder = TextEncoder(vocab)
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle') if parse_sentences else None
    if parse_sentences:
        print("Parsing sentences is activated")

    # Load the train data
    path_to_train_csv = os.path.join(path_to_data, "train.csv")
    print("Reading files in %s" % path_to_train_csv)
    # Open the csv file
    train_csv = pd.read_csv(path_to_train_csv)
    train_ids = train_csv["id"].values
    train_texts = train_csv["comment_text"]
    train_labels = train_csv[LABEL_NAMES].values

    if not parse_sentences:
        train_data = np.asarray([encoder.encode_words(tokens) for tokens in tokenize_texts(train_texts, sent_detector=None)],
                                dtype=list)
    else:
        # Filter out the 0 lenght sentences
        train_data = [[encoder.encode_words(tokens) for tokens in sent] for sent in
                                tokenize_texts(train_texts, sent_detector=sent_detector)]
        train_data = [sent for sent in train_data if len(sent) > 0]
        train_data = np.asarray(train_data, dtype=list)

    # Load the test data
    path_to_test_csv = os.path.join(path_to_data, "test.csv")
    print("Reading files in %s" % path_to_test_csv)
    # Open the csv file
    # We need to prevent "N/A" from being interpreted as a nan
    test_csv = pd.read_csv(path_to_test_csv, keep_default_na=False, na_values=[])
    test_ids = test_csv["id"].values
    test_texts = test_csv["comment_text"]

    if not parse_sentences:
        test_data = np.asarray([encoder.encode_words(tokens) for tokens in tokenize_texts(test_texts, sent_detector=None)],
                                dtype=list)
    else:
        # Filter out the 0 lenght sentences
        test_data = [[encoder.encode_words(tokens) for tokens in sent] for sent in
                      tokenize_texts(test_texts, sent_detector=sent_detector)]
        test_data = [sent for sent in test_data if len(sent) > 0]
        test_data = np.asarray(test_data, dtype=list)

    return (train_ids, train_data, train_labels), (test_ids, test_data), encoder


def load_w2v(w2v_path):
    extension = os.path.splitext(w2v_path)[1]
    print(extension)
    if extension == '.bin':
        word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    elif extension == '.w2v':
        model = Word2Vec.load(w2v_path)
        word_vectors = model.wv
    else:
        word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    return word_vectors


def embed(word_vectors, encoder):
    assert len(encoder) > 0
    num_features = word_vectors[encoder.get_word(1)].shape[0]
    # Add 1 to length for the padding
    embeddings = np.zeros((len(encoder) + 1, num_features))
    for i in range(1, len(encoder) + 1):
        embeddings[i] = word_vectors[encoder.get_word(i)]
    return embeddings


def save_data(save_dest, train_set, test_set, encoder):
    # First save the encoder
    encoder.export(os.path.join(save_dest, "word_to_id.pkl"))
    # Then save the train data
    train_ids, train_data, train_labels = train_set
    np.savez(os.path.join(save_dest, "train.npz"), ids=train_ids, data=train_data, labels=train_labels)
    # Then the test data
    test_ids, test_data = test_set
    np.savez(os.path.join(save_dest, "test.npz"), ids=test_ids, data=test_data)


def save_embeddings(save_dest, embeddings):
    np.save(os.path.join(save_dest, "embeddings.npy"), embeddings)


class TextEncoder(object):
    def __init__(self, vocab):
        self.curr_id = 1
        self.word_to_id = {"<PAD>": 0}
        self.id_to_word = {0: "<PAD>"}
        # These are the words we are allowed to use
        self.vocab = vocab

    def __getitem__(self, w):
        return self.word_to_id[w]

    def __contains__(self, w):
        return w in self.word_to_id

    def __len__(self):
        # Subtract 1 to discount the padding
        return len(self.id_to_word) - 1

    def reset_vocab(self):
        self.vocab = self.word_to_id

    def export(self, fp):
        with open(fp, 'wb') as export_file:
            pkl.dump(self.word_to_id, export_file)

    def get_word(self, word_id):
        return self.id_to_word[word_id]

    def encode(self, w):
        if w not in self:
            self.word_to_id[w] = self.curr_id
            self.id_to_word[self.curr_id] = w
            self.curr_id += 1
        # Return the id of the input word
        return self[w]

    def encode_words(self, words):
        # Return the encoded version and prune words not in the vocabulary
        # Lower case the word if its not in the vocab so we check if
        # the lowercase version is in there.
        process_words = [(w if w in self.vocab else w.lower()) for w in words]
        return [self.encode(w) for w in process_words if w in self.vocab]


if __name__ == "__main__":
    # Load, process, then save the data
    toxic_train_set, toxic_test_set, toxic_encoder = load_data(
        load_w2v(args.word2vec).vocab, path_to_data=args.data, parse_sentences=args.tokenize)
    toxic_encoder.reset_vocab()
    save_data(args.save, toxic_train_set, toxic_test_set, toxic_encoder)
    # Remove the train and test data from the memory
    del toxic_train_set
    del toxic_test_set
    # Now create and save embeddings
    save_embeddings(args.save, embed(load_w2v(args.word2vec), toxic_encoder))
