import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J

from models.abstract_model import AEmbeddingModel
from layers import build_layer
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

    def __init__(self, embeddings_name, conv_layers, fc_layers, pool, global_pool, block_size=2,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(DPCNN, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size,
                                    num_features=num_features, numpy_embeddings=numpy_embeddings)

        self.conv_layers = nn.ModuleList([Conv1d(**conv_layer) for conv_layer in conv_layers])
        self.fc_layers = nn.ModuleList([FullyConnected(**fc_layer) for fc_layer in fc_layers])
        self.pool = build_layer(**pool)
        # self.conv_layers = nn.ModuleList([nn.Conv1d(300, 300, 3, padding=1) for _ in range(14)])
        # self.fc_layers = nn.ModuleList([nn.Linear(300, 6)])
        self.pool = build_layer(**pool)
        # self.pool = nn.MaxPool1d(3, stride=2, padding=0)
        self.global_pool = build_layer(**global_pool)
        self.block_size = block_size
        self.min_len = 1

    def calc_downsize(self, input_size):
        output_size = input_size
        for i, conv_layer in enumerate(self.conv_layers):
            output_size = conv_layer.calc_output_size(output_size)
            if i and i % self.block_size == 0 and i != len(self.conv_layers):
                output_size = self.pool.calc_output_size(output_size)
        return input_size - output_size

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # If a sample is too short extend it
        x = [L.pad_numpy_to_length(sample, length=self.min_len+self.calc_downsize(len(sample))) for sample
             in x]
        # max_len = max(max(len(seq) for seq in x), 20)
        # x = np.array([L.pad_numpy_to_length(sample, length=max_len) for sample in x], dtype=int)
        # Transpose to get features x length
        return [self.embeddings(Variable(J.from_numpy(sample).long(), volatile=volatile)) for sample in x]
        # return self.embeddings(Variable(J.from_numpy(x).long(), volatile=volatile))

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        residual = x
        for i, conv_layer in enumerate(self.conv_layers):
            print(i)
            x = conv_layer(x)
            if i and i % self.block_size == 0 and i != len(self.conv_layers):
                assert all(seq_x.size() == seq_residual.size() for seq_x, seq_residual in zip(x, residual))
                x = [seq_x + seq_residual for seq_x, seq_residual in zip(x, residual)]
                x = self.pool(x)
                residual = x
        x = self.global_pool(x)
        x = L.flatten(x)  # B x F

        # Run the fc layer if we have one
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x
        return F.sigmoid(self.loss_in)

        # x = x.transpose(1, 2).contiguous()
        # residual = x
        # for i, conv_layer in enumerate(self.conv_layers):
        #     # print(i)
        #     x = conv_layer(x)
        #     if i and i % self.block_size == 0 and i != len(self.conv_layers):
        #         # print(x.size())
        #         # print(residual.size())
        #         assert x.size() == residual.size()
        #         x = residual + x
        #         x = self.pool(x)
        #         residual = x
        # x = x.transpose(1, 2).contiguous()
        # x, _ = torch.max(x, dim=1)
        #
        # for fc_layer in self.fc_layers:
        #     x = fc_layer(x)
        # self.loss_in = x
        # return F.sigmoid(self.loss_in)



registry.register_model("cnn-emb", CNNEmb)
registry.register_model("dpcnn", DPCNN)