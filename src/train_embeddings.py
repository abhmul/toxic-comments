import logging
import argparse
import os
from tqdm import tqdm

import numpy as np

from gensim.models import Word2Vec, FastText

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging

parser = argparse.ArgumentParser(description='Train embeddings for the dataset.')
parser.add_argument('-d', '--data', default="../input", help='Path to input data')
parser.add_argument('-s', '--save', required=True, help='Save path for the new word2vec model')
parser.add_argument('--size', default=300, type=int, help='Size of the vectors to train')
parser.add_argument('--other_embeddings', default="", help='Other embeddings to intersect with')
parser.add_argument('--intersect_after', action='store_true', help='Intersects the other embeddings after training.')
parser.add_argument('--min_count', default=0, type=int, help='Minimum number of times a word must occur to be ' +
                                                             'included.')
parser.add_argument('--sg', action='store_true', help='Trains the model as a skipgram model')
parser.add_argument('--embeddings_type', default='word2vec', help='The type of embeddings to train. Should be ' +
                                                                  '\'word2vec\' or \'fasttext\'.')
parser.add_argument('--intersect_type', default='word2vec', help='The type of embeddings to intersect. Should be ' +
                                                                 '\'word2vec\' or \'glove\' if training Word2Vec.' +
                                                                 '\'fasttext\' if training FastText.')
args = parser.parse_args()


def load_clean_texts(path_to_data="../processed_input/"):
    train_texts = np.load(os.path.join(path_to_data, "clean_train.npy"))
    test_texts = np.load(os.path.join(path_to_data, "clean_test.npy"))
    texts = np.concatenate([train_texts, test_texts], axis=0)
    np.random.shuffle(texts)
    return texts


def intersect_glove_format(w2v_model, glove_fname, lockf=0.0):
    overlap_count = 0
    logger.info("loading projection weights from %s", glove_fname)
    # Get the embedding dim first
    vector_size = None
    f = open(glove_fname)
    for line in f:
        values = line.split()
        vector_size = len(np.asarray(values[-300:], dtype='float32'))
        break
    f.close()

    if not vector_size == w2v_model.vector_size:
        raise ValueError("incompatible vector size %d in file %s" % (vector_size, glove_fname))

        # Now create the embeddings matrix
    f = open(glove_fname)
    for line in tqdm(f):
        values = line.split()
        word = ' '.join(values[:-300])
        weights = np.asarray(values[-300:], dtype='float32')
        if word in w2v_model.wv.vocab:
            overlap_count += 1
            w2v_model.wv.syn0[w2v_model.wv.vocab[word].index] = weights
            w2v_model.syn0_lockf[w2v_model.wv.vocab[word].index] = lockf
    f.close()
    logger.info("merged %d vectors into %s matrix from %s", overlap_count, w2v_model.wv.syn0.shape, glove_fname)


def intersect_embeddings(model, other_embeddings, lockf, intersect_type='word2vec'):
    print("Intersecting with", intersect_type, other_embeddings)
    if intersect_type == 'glove':
        intersect_glove_format(model, other_embeddings, lockf=lockf)
    if intersect_type == 'word2vec':
        binary = os.path.splitext(other_embeddings)[-1] == ".bin"
        model.intersect_word2vec_format(other_embeddings, lockf=lockf, binary=binary)
    if intersect_type == 'fasttext':
        raise NotImplementedError()
        binary = os.path.splitext(other_embeddings)[-1] == ".bin"
        model.intersect_word2vec_format(other_embeddings, lockf=lockf, binary=binary)


def train_w2v_model(datagen, other_embeddings="", intersect_type='word2vec', intersect_after=False, size=100,
                   min_count=5, sg=False):
    assert intersect_type in {'word2vec', 'glove'}
    print("Creating Word2Vec model with dimension", size)
    print("Keeping words that occur at least", min_count, "times")
    model_type = Word2Vec
    gensim_model = model_type(size=size, workers=6, min_count=min_count, sg=float(sg))
    gensim_model.build_vocab(datagen)
    if other_embeddings and not intersect_after:
        intersect_embeddings(gensim_model, other_embeddings, lockf=0.0, intersect_type=intersect_type)
    # Train the word2vec model
    gensim_model.train(datagen, total_examples=gensim_model.corpus_count, epochs=gensim_model.iter, start_alpha=gensim_model.alpha, end_alpha=gensim_model.min_alpha)
    if other_embeddings and intersect_after:
        intersect_embeddings(gensim_model, other_embeddings, lockf=0.0, intersect_type=intersect_type)
    return gensim_model


def train_fasttext_model(datagen, size=100, min_count=5, sg=False):
    print("Creating Fasttext model with dimension", size)
    print("Keeping words that occur at least", min_count, "times")
    model_type = FastText
    gensim_model = model_type(size=size, workers=6, min_count=min_count, sg=float(sg))
    gensim_model.build_vocab(datagen)
    gensim_model.train(datagen, total_examples=gensim_model.corpus_count, epochs=gensim_model.iter, start_alpha=gensim_model.alpha, end_alpha=gensim_model.min_alpha)
    return gensim_model


def train_model(datagen, other_embeddings="", embeddings_type='word2vec', intersect_type='word2vec', intersect_after=False, size=100,
                min_count=5, sg=False):
    if embeddings_type == 'word2vec':
        return train_w2v_model(datagen, other_embeddings=other_embeddings, intersect_type=intersect_type,
                               intersect_after=intersect_after, size=size, min_count=min_count, sg=sg)
    elif embeddings_type == 'fasttext':
        return train_fasttext_model(datagen, size=size, min_count=min_count, sg=sg)
    raise NotImplementedError("Cannot train %s type of model" % embeddings_type)


if __name__ == "__main__":
    datagen = load_clean_texts(args.data)
    gensim_model = train_model(datagen, other_embeddings=args.other_embeddings, embeddings_type=args.embeddings_type,
                               intersect_type=args.intersect_type, intersect_after=args.intersect_after, size=args.size,
                               min_count=args.min_count, sg=args.sg)
    # Save the trained model's vectors
    if args.embeddings_type == 'word2vec':
        gensim_model.wv.save_word2vec_format(args.save, binary=True)
    else:
        gensim_model.save(args.save)