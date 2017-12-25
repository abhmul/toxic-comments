import os
import argparse
import pickle as pkl

import numpy as np
from gensim.models.word2vec import KeyedVectors

parser = argparse.ArgumentParser(description='Construct an embeddings matrix for the data.')
parser.add_argument('-e', '--embeddings_path', required=True, help='Path to embeddings')
parser.add_argument('-w', '--word_index_path', required=True, help='Path to word_index')
parser.add_argument('-s', '--save', default="../embeddings/",
                    help='Path to directory to save the new embeddings matrix to.')
parser.add_argument('--embeddings_type', default="word2vec", help="Type of embeddings to load")
args = parser.parse_args()


def load_glove_embeddings(embeddings_path, word_index):
    # Keras starts indexing from 1
    assert len(word_index) == max(word_index.values())
    print("Reading in GloVe embeddings")
    # Get the embedding dim first
    embedding_dim = None
    f = open(embeddings_path)
    for line in f:
        values = line.split()
        embedding_dim = len(np.asarray(values[1:], dtype='float32'))
    f.close()

    # Now create the embeddings matrix
    embeddings = np.zeros((len(word_index) + 1, embedding_dim))
    not_missing = set()
    f = open(embeddings_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        if word in word_index:
            embeddings[word_index[word]] = coefs
            not_missing.add(word_index[word])
    f.close()

    # Figure out which words are missing
    missing = set(range(1, len(word_index) + 1)) - not_missing
    return embeddings, missing


def load_w2v_embeddings(embeddings_path, word_index):
    extension = os.path.splitext(embeddings_path)[1]
    is_binary = extension == ".bin"
    print("Reading in", "binary" if is_binary else "text", "Word2Vec embeddings")
    word_vectors = KeyedVectors.load_word2vec_format(embeddings_path, binary=is_binary)
    embedding_dim = word_vectors.vector_size

    # Now create the embeddings matrix
    embeddings = np.zeros((len(word_index) + 1, embedding_dim))
    missing = set()
    for word, i in word_index.items():
        if word in word_vectors.vocab:
            embeddings[i] = word_vectors[word]
        else:
            missing.add(i)
    return embeddings, missing


def load_embeddings(embeddings_path, word_index, embeddings_type="word2vec"):
    if embeddings_type == "word2vec":
        return load_w2v_embeddings(embeddings_path, word_index)
    elif embeddings_type == "glove":
        return load_glove_embeddings(embeddings_path, word_index)
    raise NotImplementedError("Embeddings type %s is not supported" % embeddings_type)


def save_embeddings(embeddings, missing, save_dir="../embeddings/"):
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
    print("Saved the embeddings")
    with open(os.path.join(save_dir, "missing.pkl"), 'wb') as missing_file:
        pkl.dump(missing, missing_file)
    print("Saved missing indicies")


if __name__ == '__main__':
    with open(args.word_index_path, 'rb') as word_index_file:
        word_index = pkl.load(word_index_file)
    embeddings_matrix, missing_indicies = load_embeddings(args.embeddings_path, word_index,
                                                          embeddings_type=args.embeddings_type)
    save_embeddings(embeddings_matrix, missing_indicies, save_dir=args.save)
