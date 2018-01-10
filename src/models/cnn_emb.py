import torch

import logging
from functools import partial

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J
import pyjet.layers as layers

from models.abstract_model import AEmbeddingModel
from layers import build_layer, build_pyjet_layer
import layers.functions as L
from layers.core import Conv1d, FullyConnected
from registry import registry


class CNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_name, conv_layers, fc_layers, pool, trainable=False, vocab_size=None, num_features=None):
        super(CNNEmb, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size, num_features=num_features)

        self.conv_layers = nn.ModuleList([Conv1d(**conv_layer) for conv_layer in conv_layers])
        self.fc_layers = nn.ModuleList([FullyConnected(**fc_layer) for fc_layer in fc_layers])
        self.pool = build_layer(**pool)
        self.min_len = 5

    def calc_downsize(self, input_size):
        output_size = input_size
        for conv_layer in self.conv_layers:
            output_size = conv_layer.calc_output_size(output_size)
        return input_size - output_size

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # If a sample is too short extend it
        x = [L.pad_numpy_to_length(sample, length=self.min_len+self.calc_downsize(len(sample))) for sample
             in x]
        # Transpose to get features x length
        return [self.embeddings(Variable(J.from_numpy(sample).long(), volatile=volatile)) for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.pool(x)
        x = L.flatten(x)  # B x F*k

        # Run the fc layer if we have one
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x
        return F.sigmoid(self.loss_in)


# TODO This has some kind of invalid pointer issue. Try out on AWS to see if the same issue crops up
# Seems to have to do with some memory issue?
class DPCNN(AEmbeddingModel):

    activations = {"relu": nn.ReLU,
                   "linear": None}

    poolings = {
        "maxpool": nn.MaxPool1d,
        "avgpool": nn.AvgPool1d,
    }

    def __init__(self, embeddings_name, conv_layers, fc_layers, pool, global_pool, block_size=2,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(DPCNN, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size,
                                    num_features=num_features, numpy_embeddings=numpy_embeddings)

        self.conv_layers = nn.ModuleList([layers.Conv1D(**conv_layer) for conv_layer in conv_layers])
        self.pool = build_pyjet_layer(**pool)
        self.fc_layers = nn.ModuleList([layers.FullyConnected(**fc_layer) for fc_layer in fc_layers])
        self.global_pool = build_pyjet_layer(**global_pool)

        self.block_size = block_size
        self.min_len = 1
        self.min_input_len = self.min_len - 1
        output_len = 0
        while output_len < self.min_len:
            self.min_input_len += 1
            output_len = self.calc_output_size(self.min_input_len)
        logging.info("BLOCK SIZE %s" % self.block_size)
        logging.info("MINIMUM INPUT LEN: %s" % self.min_input_len)

    def calc_output_size(self, input_size):
        output_size = input_size
        for i, conv_layer in enumerate(self.conv_layers):
            output_size = conv_layer.calc_output_size(output_size)
            if i and (i + 1) % self.block_size == 0 and i != len(self.conv_layers) - 1:
                output_size = self.pool.calc_output_size(output_size)
        return output_size

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # Get the maximum length of the batch to not be degenerate
        max_len = max(len(seq) for seq in x)
        max_len = max(max_len, self.min_input_len)
        x = np.array([L.pad_numpy_to_length(sample, length=max_len) for sample in x], dtype=int)
        return self.embeddings(Variable(J.from_numpy(x).long(), volatile=volatile))

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        residual = x
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i and (i+1) % self.block_size == 0 and i != len(self.conv_layers)-1:
                assert x.size() == residual.size()
                x = residual + x
                x = self.pool(x)
                residual = x
        x = x.transpose(1, 2).contiguous()
        x = self.global_pool(x)
        x = L.flatten(x)  # B x F

        # Run the fc layer if we have one
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x
        return F.sigmoid(self.loss_in)



registry.register_model("cnn-emb", CNNEmb)
registry.register_model("dpcnn", DPCNN)