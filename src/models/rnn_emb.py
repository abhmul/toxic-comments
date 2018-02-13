import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J
from models.abstract_model import AEmbeddingModel
import pyjet.layers.functions as L
from pyjet.layers import RNN, FullyConnected, Conv1D, Concatenate
from layers import build_pyjet_layer

from registry import registry


class RNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_name, rnn_layers, fc_layers, pool, resample=False,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(RNNEmb, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size,
                                     num_features=num_features, numpy_embeddings=numpy_embeddings)

        # RNN Block
        self.rnn_layers = nn.ModuleList([RNN(**rnn_layer) for rnn_layer in rnn_layers])
        # Need to work around to get backward compatibility
        if isinstance(pool, dict):
            self.pool = build_pyjet_layer(**pool)
            self.concat = None
        if isinstance(pool, list):
            self.pool = nn.ModuleList([build_pyjet_layer(**pool_i) for pool_i in pool])
            self.concat = Concatenate()
        self.use_multi_pool = self.concat is not None
        self.fc_layers = nn.ModuleList([FullyConnected(**fc_layer) for fc_layer in fc_layers])

        self.resample = resample and self.num_features != self.rnn_layers[0].input_size
        if self.resample:
            self.resampler = Conv1D(self.num_features, self.rnn_layers[0].input_size, 1, use_bias=False)

        # For testing purposes only
        # self.att = AttentionHierarchy(self.num_features, 300, encoder_dropout=0.25, att_type='linear')
        self.min_len = 1

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # Get the seq lens and pad it
        seq_lens = [max(len(sample), self.min_len) for sample in x]
        x = np.array([L.pad_numpy_to_length(sample, length=max(seq_lens)) for sample in x], dtype=int)
        return self.embeddings(Variable(J.from_numpy(x).long(), volatile=volatile)), seq_lens

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        x, seq_lens = x
        # Apply the resampler if necessary
        if self.resample:
            x = self.resampler(x)
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)  # B x Li x H

        # Apply the mask
        x = L.unpad_sequences(x, seq_lens)
        # Do the pooling
        if self.use_multi_pool:
            x = self.concat([pool_i(x) for pool_i in self.pool])
        else:
            x = self.pools[0](x)
        x = L.flatten(x)  # B x k*H
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x  # B x 6
        return F.sigmoid(self.loss_in)

    def reset_parameters(self):
        for layer in self.rnn_layers:
            layer.reset_parameters()
        if self.use_multi_pool:
            for pool in self.pool:
                pool.reset_parameters()
        else:
            self.pool.reset_parameters()
        for layer in self.fc_layers:
            layer.reset_parameters()
        if self.resample:
            self.resampler.reset_parameters()


registry.register_model("rnn-emb", RNNEmb)
