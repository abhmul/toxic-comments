import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J
from models.abstract_model import AEmbeddingModel
import layers.functions as L
from layers import RNN, FullyConnected, build_layer

from registry import registry


class RNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_name, rnn_layers, fc_layers, pool,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(RNNEmb, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size,
                                     num_features=num_features, numpy_embeddings=numpy_embeddings)

        # RNN Block
        self.rnn_layers = nn.ModuleList([RNN(**rnn_layer) for rnn_layer in rnn_layers])
        self.pool = build_layer(**pool)
        self.fc_layers = nn.ModuleList([FullyConnected(**fc_layer) for fc_layer in fc_layers])

        # For testing purposes only
        # self.att = AttentionHierarchy(self.num_features, 300, encoder_dropout=0.25, att_type='linear')
        self.min_len = 1

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # If a sample is too short extend it
        x = [L.pad_numpy_to_length(sample, length=self.min_len) for sample in x]
        # Transpose to get features x length
        return [self.embeddings(Variable(J.from_numpy(sample).long(), volatile=volatile)) for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)  # B x Li x H
        x = self.pool(x)
        # For testing purposes only
        # x = self.att(x)
        x = L.flatten(x)  # B x k*H

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x  # B x 6
        return F.sigmoid(self.loss_in)


class DPRNN(AEmbeddingModel):

    def __init__(self, embeddings_name, rnn_layers, fc_layers, pool, global_pool, block_size=1,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(DPRNN, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size,
                                    num_features=num_features, numpy_embeddings=numpy_embeddings)

        for rnn_layer in rnn_layers:
            rnn_layer["n_layers"] = block_size
        self.rnn_layers = nn.ModuleList([RNN(**rnn_layer) for rnn_layer in rnn_layers])
        self.pool = build_layer(**pool)
        self.fc_layers = nn.ModuleList([FullyConnected(**fc_layer) for fc_layer in fc_layers])
        self.global_pool = build_layer(**global_pool)
        self.block_size = block_size
        self.min_len = 1

    def calc_downsize(self, input_size):
        output_size = input_size
        for i, rnn_layer in enumerate(self.rnn_layers):
            if i and i % self.block_size == 0 and i != len(self.rnn_layers):
                output_size = self.pool.calc_output_size(output_size)
        return input_size - output_size

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # If a sample is too short extend it
        x = [L.pad_numpy_to_length(sample, length=20) for sample in x]
        # Transpose to get features x length
        return [self.embeddings(Variable(J.from_numpy(sample).long(), volatile=volatile)) for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        residual = x
        for i, rnn_layer in enumerate(self.rnn_layers):
            # print(i)
            x = rnn_layer(x)
            if (i + 1) % self.block_size == 0:
                assert all(seq_x.size() == seq_residual.size() for seq_x, seq_residual in zip(x, residual))
                x = [seq_x + seq_residual for seq_x, seq_residual in zip(x, residual)]
                if i != len(self.rnn_layers) - 1:
                    x = self.pool(x)
                    residual = x
        x = self.global_pool(x)
        x = L.flatten(x)  # B x F

        # Run the fc layer if we have one
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x
        return F.sigmoid(self.loss_in)


registry.register_model("rnn-emb", RNNEmb)
registry.register_model("dprnn", DPRNN)
