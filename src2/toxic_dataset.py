from pyjet.data import NpDataset
import numpy as np
import pickle as pkl
import pandas as pd

LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class ToxicData(object):

    def __init__(self, train_path, test_path, word_index_path=""):
        self.train_path = train_path
        self.test_path = test_path
        self.word_index = word_index_path

    @staticmethod
    def load_supervised(data):
        ids = data["ids"]
        text = data["texts"]
        if "labels" in data:
            labels = data["labels"]
        else:
            labels = None
        return ids, NpDataset(text, labels)

    @staticmethod
    def save_submission(submission_fname, pred_ids, predictions):
        submid = pd.DataFrame({'id': pred_ids})
        submission = pd.concat([submid, pd.DataFrame(data=predictions, columns=LABEL_NAMES)], axis=1)
        submission.to_csv(submission_fname, index=False)

    def load_train_supervised(self):
        return self.load_supervised(np.load(self.train_path))

    def load_train(self, mode="sup"):
        if mode == "sup":
            return self.load_train_supervised()
        else:
            raise NotImplementedError()

    def load_test(self):
        return self.load_supervised(np.load(self.test_path))

    def load_dictionary(self):
        with open(self.word_index, "rb") as mapping:
            word_index = pkl.load(mapping)
        vocab = [None]*len(word_index)
        vocab[0] = "<PAD>"
        for word, index in word_index.items():
            vocab[index] = word
        return word_index, vocab