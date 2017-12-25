import os
import pickle as pkl

import torch.nn as nn

import numpy as np

from pyjet.models import SLModel
import pyjet.backend as J


class AEmbeddingModel(SLModel):

    def __init__(self, embeddings_path="", trainable=False, vocab_size=None, num_features=None):
        super(AEmbeddingModel, self).__init__()
        self.embeddings_path = embeddings_path
        self.embeddings, self.missing = self.load_embeddings_path(embeddings_path)
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
        embeddings.weight.requires_grad = trainable
        with open(os.path.join(embeddings_path, "missing.pkl"), 'rb') as missing_file:
            missing = pkl.load(missing_file)
        return embeddings, missing

    def forward(self, *inputs, **kwargs):
        NotImplementedError()