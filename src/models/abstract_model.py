import os
import pickle as pkl

import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from pyjet.models import SLModel
import pyjet.backend as J


def fix_embeddings_name(embeddings_name):
    return os.path.join("..", "embeddings", embeddings_name + "/")


class NpEmbeddings(np.ndarray):

    @property
    def num_embeddings(self):
        return self.shape[0]

    @property
    def embedding_dim(self):
        return self.shape[1]

    def __call__(self, indicies):
        return Variable(J.from_numpy(self[J.to_numpy(indicies.data)].astype(np.float32)),
                        volatile=indicies.volatile)


class AEmbeddingModel(SLModel):

    def __init__(self, embeddings_name="", trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(AEmbeddingModel, self).__init__()
        self.embeddings_path = fix_embeddings_name(embeddings_name)
        self.numpy_embeddings = numpy_embeddings
        self.embeddings, self.missing = self.load_embeddings_path(self.embeddings_path, trainable=trainable,
                                                                  vocab_size=vocab_size, num_features=num_features,
                                                                  numpy_embeddings=numpy_embeddings)
        self.sgd_params = {id(param) for param in self.embeddings.parameters()}
        self.num_features = self.embeddings.embedding_dim
        self.vocab_size = self.embeddings.num_embeddings

    @staticmethod
    def load_embeddings_path(embeddings_path="", trainable=False, vocab_size=None, num_features=None,
                             numpy_embeddings=False):
        # Load the embeddings file
        if embeddings_path:
            np_embeddings = np.load(os.path.join(embeddings_path, "embeddings.npy"))
        else:
            assert vocab_size is not None or num_features is not None
            np_embeddings = np.zeros((vocab_size + 1, num_features))

        # Create the embeddings layer
        if not numpy_embeddings:
            embeddings = nn.Embedding(*np_embeddings.shape, padding_idx=0, scale_grad_by_freq=False, sparse=True)
            embeddings.weight.data.copy_(J.from_numpy(np_embeddings))
            print("Trainable Embeddings: ", trainable)
            embeddings.weight.requires_grad = trainable
        else:
            embeddings = np_embeddings.view(NpEmbeddings)

        # Load the missing if we are not training them
        if trainable:
            missing = set()
        else:
            with open(os.path.join(embeddings_path, "missing.pkl"), 'rb') as missing_file:
                missing = pkl.load(missing_file)

        print("Loaded", embeddings.num_embeddings, "embeddings and", len(missing), "missing words.")
        return embeddings, missing

    def forward(self, *inputs, **kwargs):
        NotImplementedError()

    def trainable_params(self, sgd=False):
        if sgd:
            return [param for param in self.parameters() if id(param) in self.sgd_params and param.requires_grad]
        else:
            return [param for param in self.parameters() if (id(param) not in self.sgd_params) and param.requires_grad]
