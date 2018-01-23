from pyjet.data import NpDataset, Dataset
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


class MultiNpDatasetAugmenter(Dataset):

    def __init__(self, original_dataset: NpDataset, *datasets: NpDataset):
        super().__init__()
        assert all(len(dataset) == len(original_dataset) for dataset in datasets), "Datasets must be of same length"
        assert all(dataset.output_labels == original_dataset.output_labels for dataset in
                   datasets), "Some datasets output labels while others do not."
        self.datasets = [original_dataset] + datasets
        self.original_dataset = original_dataset
        self.n_datasets = len(self.datasets)
        self.output_labels = original_dataset.output_labels
        self.__length = len(original_dataset)

    def __len__(self):
        return self.__length

    def create_batch(self, batch_indicies):
        # Randomly choose a dataset for each index
        dataset_inds = np.random.randint(self.n_datasets, size=len(batch_indicies))
        x, y = None, None
        # Create each sub-batch for each sampled dataset
        for i, dataset in enumerate(self.datasets):
            is_dataset = [dataset_inds == i]
            subbatch = dataset.create_batch(batch_indicies[is_dataset])
            # Split the dataset if it has labels with it
            if self.output_labels:
                xsub, ysub = subbatch
                # If we haven't initialized the label batch, use the subbatch to infer the size
                if y is None:
                    y = np.empty((len(batch_indicies)) + ysub.shape[1:], dtype=ysub.dtype)
                y[is_dataset] = ysub
            else:
                xsub = subbatch

            if x is None:
                x = np.empty((len(batch_indicies)) + xsub.shape[1:], dtype=xsub.dtype)
            x[is_dataset] = xsub
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


