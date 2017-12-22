import os
import argparse
import pandas as pd
import numpy as np

from gensim.models.word2vec import Word2Vec

parser = argparse.ArgumentParser(description='Train embeddings for the dataset.')
parser.add_argument('-d', '--data', default="../input", help='Path to input data')
parser.add_argument('-s', '--save', required=True, help='Save path for the new word2vec model')
parser.add_argument('--size', default=100, type=int, help='Size of the vectors to train')
parser.add_argument('--other_embeddings', default="", help='Other embeddings to intersect with')
parser.add_argument('--intersect_after', action='store_true', help='Intersects the other embeddings after training.')
parser.add_argument('--min_count', default=5, type=int, help='Minimum number of times a word must occur to be ' +
                                                             'included.')
args = parser.parse_args()


def load_data(path_to_data="../input"):
    # Get the train csv
    path_to_train_csv = os.path.join(path_to_data, "train.csv")
    print("Reading files in %s" % path_to_train_csv)
    # Open the csv file
    train_texts = pd.read_csv(path_to_train_csv)["comment_text"]
    # Get the test csv
    path_to_test_csv = os.path.join(path_to_data, "test.csv")
    print("Reading files in %s" % path_to_test_csv)
    test_texts = pd.read_csv(path_to_test_csv, keep_default_na=False, na_values=[])["comment_text"]
    # Combined texts
    all_texts = pd.concat([train_texts, test_texts]).values
    # Shuffle the texts
    np.random.shuffle(all_texts)
    return all_texts


def train_word2vec(datagen, other_embeddings="", intersect_after=False, size=100, min_count=5):
    w2v = Word2Vec(size=size, workers=4, min_count=min_count)
    w2v.build_vocab(datagen)
    if other_embeddings and not intersect_after:
        binary = os.path.splitext(other_embeddings)[-1] == ".bin"
        w2v.intersect_word2vec_format(other_embeddings, lockf=1.0, binary=binary)
    # Train the word2vec model
    w2v.train(datagen, total_examples=w2v.corpus_count, epochs=w2v.iter, start_alpha=w2v.alpha, end_alpha=w2v.min_alpha)
    if other_embeddings and intersect_after:
        binary = os.path.splitext(other_embeddings)[-1] == ".bin"
        w2v.intersect_word2vec_format(other_embeddings, lockf=0.0, binary=binary)
    return w2v


if __name__ == "__main__":
    datagen = load_data(args.path_to_data)
    w2v_model = train_word2vec(datagen, other_embeddings=args.other_embeddings, intersect_after=args.intersect_after,
                               size=args.size, min_count=args.min_count)
    # Save the trained model's vectors
    w2v_model.save_word2vec_format(args.save, binary=True)

