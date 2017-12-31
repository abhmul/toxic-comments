import os
import pickle as pkl

import torch.nn as nn

import numpy as np

from pyjet.models import SLModel
import pyjet.backend as J


def fix_embeddings_name(embeddings_name):
    return os.path.join("..", "embeddings", embeddings_name + "/")


class AEmbeddingModel(SLModel):

    def __init__(self, embeddings_name="", trainable=False, vocab_size=None, num_features=None):
        super(AEmbeddingModel, self).__init__()
        self.embeddings_path = fix_embeddings_name(embeddings_name)
        self.embeddings, self.missing = self.load_embeddings_path(self.embeddings_path, trainable=trainable,
                                                                  vocab_size=vocab_size, num_features=num_features)
        self.num_features = self.embeddings.embedding_dim
        self.vocab_size = self.embeddings.num_embeddings

    @staticmethod
    def load_embeddings_path(embeddings_path="", trainable=False, vocab_size=None, num_features=None):
        if embeddings_path:
            np_embeddings = np.load(os.path.join(embeddings_path, "embeddings.npy"))
        else:
            assert vocab_size is not None or num_features is not None
            np_embeddings = np.zeros((vocab_size + 1, num_features))
        embeddings = nn.Embedding(*np_embeddings.shape, padding_idx=0, scale_grad_by_freq=True)
        embeddings.weight.data.copy_(J.from_numpy(np_embeddings))
        print("Trainable Embeddings: ", trainable)
        embeddings.weight.requires_grad = trainable
        with open(os.path.join(embeddings_path, "missing.pkl"), 'rb') as missing_file:
            missing = pkl.load(missing_file)
        print("Loaded", embeddings.num_embeddings, "embeddings and", len(missing), "missing words.")
        return embeddings, missing

    def forward(self, *inputs, **kwargs):
        NotImplementedError()
