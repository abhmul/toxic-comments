import glob
from pyjet.data import NpDataset, Dataset
import numpy as np
import pickle as pkl
import logging
import pandas as pd

LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class ToxicData(object):

    def __init__(self, train_path, test_path, word_index_path="", augmented_path="", original_prob=None, fixed_len=None):
        self.train_path = train_path
        self.test_path = test_path
        self.augmented = bool(augmented_path)
        self.word_index = word_index_path
        # Glob the augmented datasets
        if self.augmented:
            self.augmented_paths = glob.glob(augmented_path)
        self.original_prob = original_prob
        if fixed_len is None or fixed_len < 0:
            self.fixed_len = None
        else:
            self.fixed_len = fixed_len


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
        ids, original_dataset = self.load_supervised(np.load(self.train_path))
        if self.augmented:
            aug_datasets = []
            for aug_path in self.augmented_paths:
                logging.info('Loading augmented dataset from %s' % aug_path)
                _, aug_dataset = self.load_supervised(np.load(aug_path))
                aug_datasets.append(aug_dataset)
            return ids, MultiNpDatasetAugmenter(original_dataset, *aug_datasets, use_original_prob=self.original_prob,
                                                fixed_len=self.fixed_len)
        return ids, original_dataset

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


class MultiNpDatasetAugmenter(Dataset):

    def __init__(self, original_dataset: NpDataset, *datasets: NpDataset, use_original_prob=None, fixed_len=None):
        super().__init__()
        assert all(len(dataset) == len(original_dataset) for dataset in datasets), "Datasets must be of same length"
        assert all(dataset.output_labels == original_dataset.output_labels for dataset in
                   datasets), "Some datasets output labels while others do not."
        self.datasets = [original_dataset] + list(datasets)
        self.original_dataset = original_dataset
        self.n_datasets = len(self.datasets)
        self.output_labels = original_dataset.output_labels
        self.use_original_prob = use_original_prob
        self.fixed_len = fixed_len
        logging.info("Using a fixed len of %s" % self.fixed_len)
        self.__length = len(original_dataset)
        self.__dataset_inds = np.arange(self.n_datasets, dtype=int)
        if self.use_original_prob is None:
            self.__sample_probs = None
        else:
            self.__sample_probs = np.array(
                [self.use_original_prob] + [(1. - self.use_original_prob) / (self.n_datasets - 1.)] * (
                 self.n_datasets - 1))

    def __len__(self):
        return self.__length

    def create_batch(self, batch_indicies):
        # Randomly choose a dataset for each index where the original has 50% chance of being used
        dataset_inds = np.random.choice(self.__dataset_inds, size=len(batch_indicies), p=self.__sample_probs)
        x, y = None, None
        # Create each sub-batch for each sampled dataset
        for i, dataset in enumerate(self.datasets):
            is_dataset = dataset_inds == i
            subbatch_inds = batch_indicies[is_dataset]
            # Don't do anything if there are no samples from this dataset
            if len(subbatch_inds) == 0:
                continue
            subbatch = dataset.create_batch(subbatch_inds)
            # logging.info("Dataset %s subbatch is of length %s = %s" % (i, len(subbatch[0]), len(subbatch_inds)))
            # Split the dataset if it has labels with it
            if self.output_labels:
                xsub, ysub = subbatch
                # If we haven't initialized the label batch, use the subbatch to infer the size
                if y is None:
                    y = np.empty((len(batch_indicies),) + ysub.shape[1:], dtype=ysub.dtype)
                y[is_dataset] = ysub
            else:
                xsub = subbatch

            if x is None:
                x = np.empty((len(batch_indicies),) + xsub.shape[1:], dtype=xsub.dtype)
            x[is_dataset] = xsub

        # Do fixed length augmentation if so
        if self.fixed_len is not None:
            x_new = np.empty((len(x), self.fixed_len), dtype=int)
            for i, sample in enumerate(x):
                end_start = len(sample) - self.fixed_len + 1
                if end_start <= 1:
                    x_new[i, :len(sample)] = np.array(sample)
                start = np.random.randint(0, end_start)
                x_new[i] = np.array(sample[start: start + self.fixed_len])
            x = x_new

        # Yield the final output
        if self.output_labels:
            return x, y
        return x

    def validation_split(self, split=0.2, shuffle=False, seed=None, stratified=False):
        """
        NOTE: Only use stratified if the labels are the same between the augmented sets
        This will assume the first dataset provided is the original and others are augmented
        versions. Thus the validation set will only be pulled from the original dataset.
        """
        # Get the split indicies
        train_split, val_split = self.original_dataset.get_split_indicies(split, shuffle, seed, stratified)
        # Create each subdataset
        train_data = MultiNpDatasetAugmenter(
            *(NpDataset(dataset.x[train_split], None if not self.output_labels else dataset.y[train_split]) for dataset
              in self.datasets))
        # We use the original dataset for the validation set
        val_data = NpDataset(self.original_dataset.x[val_split],
                             None if not self.output_labels else self.original_dataset.y[val_split])
        return train_data, val_data

    def kfold(self, k=True, shuffle=False, seed=None):
        for train_split_a, train_split_b, val_split in self.original_dataset.get_kfold_indices(k, shuffle, seed):
            train_data = MultiNpDatasetAugmenter(
                *(NpDataset(np.concatenate([dataset.x[train_split_a], dataset.x[train_split_b]]),
                          None if not self.output_labels else np.concatenate(
                              [dataset.y[train_split_a], dataset.y[train_split_b]])) for dataset in self.datasets)
            )

            val_data = NpDataset(self.original_dataset.x[val_split],
                                 None if not self.output_labels else self.original_dataset.y[val_split])
            yield train_data, val_data


