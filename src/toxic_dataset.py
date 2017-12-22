from pyjet.data import NpDataset, Dataset
import numpy as np
import pickle as pkl


class ToxicData(object):

    def __init__(self, train_path, test_path, embeddings_path="", word_to_id_path=""):
        self.train_path = train_path
        self.test_path = test_path
        self.embeddings_path = embeddings_path
        self.word_to_id_path = word_to_id_path

    @staticmethod
    def load_supervised(data):
        ids = data["ids"]
        text = data["data"]
        labels = data["labels"]
        return ids, NpDataset(text, labels)

    def load_train_supervised(self):
        return self.load_supervised(np.load(self.train_path))

    def load_train(self, mode="sup"):
        if mode == "sup":
            return self.load_train_supervised()
        else:
            raise NotImplementedError()

    def load_test(self):
        return self.load_supervised(np.load(self.test_path))

    def embed(self, dataset, parsed_sent=False, logger=None):
        assert self.embeddings_path, "Cannot embed without an embeddings path."
        embeddings = np.load(self.embeddings_path)
        return EmbeddingDataset(embeddings, dataset, parsed_sent=parsed_sent, logger=logger)

    def load_dictionary(self):
        with open(self.word_to_id_path, "rb") as mapping:
            word_to_id = pkl.load(mapping)
        id_to_word = {word_to_id[w]: w for w in word_to_id}
        return word_to_id, id_to_word


class EmbeddingDataset(Dataset):

    def __init__(self, embeddings, dataset, parsed_sent=False, logger=None):
        self.embeddings = embeddings
        self.num_features = self.embeddings.shape[1]
        self.dataset = dataset
        self.parsed_sent = parsed_sent
        self.logger = logger
        print(self.logger)
        try:
            self.output_labels = self.dataset.output_labels
        except AttributeError:
            self.output_labels = False

    def __len__(self):
        return len(self.dataset)

    def create_batch(self, batch_indicies):
        # Grab the batch from the underlying dataset
        x_batch = self.dataset.create_batch(batch_indicies)
        if self.logger is not None:
            self.logger.register_input(
                x_batch[0] if self.output_labels else x_batch)
        # Split up the labels and samples if necessary
        if self.output_labels:
            x_batch, y_batch = x_batch

        # transpose so it is channels x words
        if self.parsed_sent:
            # Each sample is a list of embedded sentences sorted by length
            x_batch = [[self.embeddings[sent] for sent in sample if len(sent) > 0]
                       for sample in x_batch]
        else:
            # Each batch is a list of docs sorted by length
            x_batch = [self.embeddings[sample] for sample in x_batch]
        # Return the batch
        return x_batch, y_batch if self.output_labels else x_batch

    def validation_split(self, split=0.2, shuffle=True, seed=None):
        train_dataset, val_dataset = self.dataset.validation_split(
            split, shuffle, seed)
        train_embed_dataset = EmbeddingDataset(
            self.embeddings, train_dataset, parsed_sent=self.parsed_sent, logger=self.logger)
        val_embed_dataset = EmbeddingDataset(
            self.embeddings, val_dataset, parsed_sent=self.parsed_sent, logger=self.logger)
        return train_embed_dataset, val_embed_dataset