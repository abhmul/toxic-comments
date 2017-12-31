import numpy as np
import pickle as pkl
import os
import argparse

parser = argparse.ArgumentParser(description='Port over old data.')
parser.add_argument('-d', '--data', required=True, help='Path to the old processed data')
parser.add_argument('-s', '--save', required=True, help='Path to the new ported processed data.')
parser.add_argument('-e', '--embeddings', required=True, help='Path to the new ported embeddings.')


def load_old_dataset(old_dataset_path):
    old_dataset = np.load(old_dataset_path)
    ids = old_dataset["ids"]
    texts = old_dataset["data"]
    if "labels" in old_dataset:
        labels = old_dataset["labels"]
        return ids, texts, labels
    else:
        return ids, texts


def load_old_embeddings(old_embeddings_path):
    old_embeddings = np.load(old_embeddings_path)
    missing = set()
    return old_embeddings, missing


def load_old_word_index(old_word_index_path):
    with open(old_word_index_path, 'rb') as old_word_index_file:
        old_word_index = pkl.load(old_word_index_file)
    return old_word_index


def save_new_dataset(new_dataset_path, ids, texts, labels=None):
    if labels is not None:
        np.savez(new_dataset_path, ids=ids, texts=texts, labels=labels)
    else:
        np.savez(new_dataset_path, ids=ids, texts=texts)


def save_new_embeddings(new_embeddings_dir, embeddings, missing):
    np.save(os.path.join(new_embeddings_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(new_embeddings_dir, "missing.pkl"), 'wb') as missing_file:
        pkl.dump(missing, missing_file)


def save_new_word_index(new_word_index_path, word_index):
    with open(new_word_index_path, 'wb') as word_index_file:
        pkl.dump(word_index, word_index_file)


def convert_tokens(old_texts):
    # Add 1 to all the text tokens
    is_sent_parsed = isinstance(old_texts[0][0], list)
    if is_sent_parsed:
        for doc in old_texts:
            for sent in doc:
                for i in range(len(sent)):
                    sent[i] += 1
    else:
        for doc in old_texts:
            for i in range(len(doc)):
                doc[i] += 1


def convert_word_index(old_word_index):
    for w in old_word_index.keys():
        old_word_index[w] += 1

def port_old_data(old_dataset_dir, save_dataset_dir, save_embeddings_dir):
    # Load the train and test data
    train_ids, train_texts, train_labels = load_old_dataset(os.path.join(old_dataset_dir, "train.npz"))
    test_ids, test_texts = load_old_dataset(os.path.join(old_dataset_dir, "test.npz"))
    # Load the old embeddings
    embeddings, missing = load_old_embeddings(os.path.join(old_dataset_dir, "embeddings.npy"))
    # Load the old word_index
    word_index = load_old_word_index(os.path.join(old_dataset_dir, "word_to_id.pkl"))

    # Convert everything
    convert_tokens(train_texts)
    convert_tokens(test_texts)
    convert_word_index(word_index)

    # Save the data in the new format
    save_new_dataset(os.path.join(save_dataset_dir, "train.npz"), train_ids, train_texts, train_labels)
    save_new_dataset(os.path.join(save_dataset_dir, "test.npz"), test_ids, test_texts)
    save_new_word_index(os.path.join(save_dataset_dir, "word_index.pkl"), word_index)
    # Save the embeddings
    save_new_embeddings(save_embeddings_dir, embeddings, missing)


if __name__ == "__main__":
    args = parser.parse_args()
    port_old_data(args.data, args.save, args.embeddings)