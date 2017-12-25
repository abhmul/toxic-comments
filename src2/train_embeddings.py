import logging
import argparse
import os

import numpy as np

from gensim.models.word2vec import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Train embeddings for the dataset.')
parser.add_argument('-d', '--data', default="../input", help='Path to input data')
parser.add_argument('-s', '--save', required=True, help='Save path for the new word2vec model')
parser.add_argument('--size', default=300, type=int, help='Size of the vectors to train')
parser.add_argument('--other_embeddings', default="", help='Other embeddings to intersect with')
parser.add_argument('--intersect_after', action='store_true', help='Intersects the other embeddings after training.')
parser.add_argument('--min_count', default=0, type=int, help='Minimum number of times a word must occur to be ' +
                                                             'included.')
args = parser.parse_args()


def load_clean_texts(path_to_data="../processed_input/"):
    train_texts = np.load(os.path.join(path_to_data, "clean_train.npy"))
    test_texts = np.load(os.path.join(path_to_data, "clean_test.npy"))
    texts = np.concatenate([train_texts, test_texts], axis=0)
    np.random.shuffle(texts)
    return texts


def train_word2vec(datagen, other_embeddings="", intersect_after=False, size=100, min_count=5):
    print("Creating Word2Vec model with dimension", size)
    print("Keeping words that occur at least", min_count, "times")
    w2v = Word2Vec(size=size, workers=4, min_count=min_count)
    w2v.build_vocab(datagen)
    if other_embeddings and not intersect_after:
        print("Intersecting before with", other_embeddings)
        binary = os.path.splitext(other_embeddings)[-1] == ".bin"
        w2v.intersect_word2vec_format(other_embeddings, lockf=1.0, binary=binary)
    # Train the word2vec model
    w2v.train(datagen, total_examples=w2v.corpus_count, epochs=w2v.iter, start_alpha=w2v.alpha, end_alpha=w2v.min_alpha)
    if other_embeddings and intersect_after:
        print("Intersecting after with", other_embeddings)
        binary = os.path.splitext(other_embeddings)[-1] == ".bin"
        w2v.intersect_word2vec_format(other_embeddings, lockf=0.0, binary=binary)
    return w2v


if __name__ == "__main__":
    datagen = load_clean_texts(args.data)
    w2v_model = train_word2vec(datagen, other_embeddings=args.other_embeddings, intersect_after=args.intersect_after,
                               size=args.size, min_count=args.min_count)
    # Save the trained model's vectors
    w2v_model.wv.save_word2vec_format(args.save, binary=True)