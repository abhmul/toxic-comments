import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J
from pyjet.layers import RNN, FullyConnected, Conv1D, MaskedInput
from models.abstract_model import AEmbeddingModel
import layers.functions as L
from layers import build_pyjet_layer

from registry import registry


class DenseRNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_name, rnn_layers, fc_layers, pool, resample=False,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(DenseRNNEmb, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size,
                                          num_features=num_features, numpy_embeddings=numpy_embeddings)

        # RNN Block
        self.rnn_layers = nn.ModuleList([RNN(**rnn_layer) for rnn_layer in rnn_layers])
        self.pool = build_pyjet_layer(**pool)
        # Smartly figure out how to mask the input
        self.mask = MaskedInput(mask_value=(-float('inf') if 'max' in self.pool else 0.))
        self.fc_layers = nn.ModuleList([FullyConnected(**fc_layer) for fc_layer in fc_layers])
        # Create a resample conv layer if the number of features does not match the input rnn
        self.resample = resample and self.num_features != self.rnn_layers[0].input_size
        if self.resample:
            self.resampler = Conv1D(self.num_features, self.rnn_layers[0].input_size, 1, use_bias=False)

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # Get the seq lens and pad it
        seq_lens = [len(sample) for sample in x]
        x = np.array([L.pad_numpy_to_length(sample, length=max(seq_lens)) for sample in x], dtype=int)
        return self.embeddings(Variable(J.from_numpy(x).long(), volatile=volatile)), seq_lens

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        x, seq_lens = x
        # Apply the resampler if necessary
        if self.resample:
            x = self.resampler(x)
        # Run the rnns
        for rnn_layer in self.rnn_layers[:-1]:
            x = torch.cat([x, rnn_layer(x)], dim=-1)  # B x L x (Hx + Hout)
        # Final rnn
        x = self.rnn_layers[-1](x)
        # Apply the mask
        self.mask(x)
        # Run the pooling and flatten
        x = self.pool(x)
        x = L.flatten(x)  # B x k*H

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x  # B x 6
        return F.sigmoid(self.loss_in)


registry.register_model("dense-rnn-emb", DenseRNNEmb)
