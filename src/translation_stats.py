import argparse
import os
import pandas as pd
from tqdm import tqdm
from scipy import spatial

from gensim.models import KeyedVectors, FastText
from Bio import pairwise2

NAN_WORD = "NA"
GAP_CHAR = "-+-"

def get_alignment_stats(alignments):
    # Pick the optimal alignment (the one that's shortest)
    alignments = [(align[0], align[1]) for align in alignments]
    scores = [align[2] for align in alignments]
    assert all(len(a1) == len(a2) for a1, a2 in alignments)
    i = sorted(range(len(alignments)), key=lambda j: len(alignments[j][0]))[0]
    alignment = alignments[i]

    # Number of word swaps is our score
    word_swaps = sum(a != b for a, b in zip(*alignment) if a != GAP_CHAR and b != GAP_CHAR)
    # Number of additions is the number of gaps in the 2nd sequence
    additions = alignment[1].count(GAP_CHAR)
    # Number of deletions is the number of gaps in the 1st sequence
    deletions = alignment[0].count(GAP_CHAR)

    return word_swaps, additions, deletions


def match_function(word_1, word_2, word_vectors):
    if word_1 not in word_vectors or word_2 not in word_vectors:
        return -1
    embedding_1 = word_vectors[word_1]
    embedding_2 = word_vectors[word_2]

    return 1 - spatial.distance.cosine(embedding_1, embedding_2)


def main():
    parser = argparse.ArgumentParser("Script for extending train dataset")
    parser.add_argument("embeddings_path")
    parser.add_argument("train_file_path")
    parser.add_argument("translated_file_glob")
    # parser.add_argument("--thread-count", type=int, default=8)
    # parser.add_argument("--result-path", default="extended_data")

    args = parser.parse_args()

    try:
        word_vectors = FastText.load_fasttext_format(args.embeddings_path)
    except NotImplementedError:
        word_vectors = FastText.load(args.embeddings_path)

    train_data = pd.read_csv(args.train_file_path)
    comments_list = train_data["comment_text"].fillna(NAN_WORD).values
