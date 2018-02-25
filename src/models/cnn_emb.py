import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import pyjet.backend as J
import pyjet.layers as layers

from models.abstract_model import AEmbeddingModel
from layers import build_pyjet_layer
import pyjet.layers.functions as L
from pyjet.layers import Conv1D, FullyConnected, Identity, SequenceConv1D
from registry import registry


class CNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_name, layers, fc_layers, pool, trainable=False, vocab_size=None, num_features=None):
        super(CNNEmb, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size, num_features=num_features)

        # Build the conv layers
        self.conv_layers = nn.Sequential()
        for i, layer in enumerate(layers):
            self.conv_layers.add_module(name=layer["name"] + str(i), module=build_pyjet_layer(**layer))

        # Build the FC layers
        self.fc_layers = nn.Sequential(*[FullyConnected(**fc_layer) for fc_layer in fc_layers])
        # Build the pooling layer
        self.pool = build_pyjet_layer(**pool)
        self.min_len = 1 if "k" not in pool else pool["k"]

    def calc_input_size(self, output_size):
        for layer in reversed(self.conv_layers):
            output_size = layer.calc_input_size(output_size)
        return output_size

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # Get the maximum length of the batch to not be degenerate
        max_len = max(len(seq) for seq in x)
        max_len = max(max_len, self.calc_input_size(self.min_len))
        x = np.array([L.pad_numpy_to_length(sample, length=max_len) for sample in x], dtype=int)
        return self.embeddings(Variable(J.from_numpy(x).long(), volatile=volatile))

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = L.flatten(x)  # B x F*k

        # Run the fc layer if we have one
        x = self.fc_layers(x)
        self.loss_in = x
        return self.loss_in

    def reset_parameters(self):
        for layer in self.conv_layers:
            layer.reset_parameters()
        self.pool.reset_parameters()
        for layer in self.fc_layers:
            layer.reset_parameters()


class ResidualBlock(nn.Module):

    def __init__(self, residual_layers, nonresidual_layers):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential()
        # Add the residual layers
        for i, layer in enumerate(residual_layers):
            self.residual.add_module(name=layer["name"] + str(i), module=build_pyjet_layer(**layer))
        # Add the nonresidual layer last
        self.nonresidual = nn.Sequential()
        # Add the residual layers
        for i, layer in enumerate(nonresidual_layers):
            self.nonresidual.add_module(name=layer["name"] + str(i), module=build_pyjet_layer(**layer))

        if len(self.residual) and self.residual[0].input_size != self.residual[-1].output_size:
            self.shortcut = SequenceConv1D(self.residual[0].input_size, self.residual[-1].output_size, 1)
        else:
            self.shortcut = Identity

    def calc_input_size(self, output_size):
        for layer in reversed(self.residual):
            output_size = layer.calc_input_size(output_size)
        for layer in reversed(self.nonresidual):
            output_size = layer.calc_input_size(output_size)
        return output_size

    def forward(self, x):
        input_x = x
        if len(self.residual):
            x = self.residual(x)
        x = [res + sample for res, sample in zip(self.shortcut(input_x), x)]
        if len(self.nonresidual):
            x = self.nonresidual(x)
        return x

    def reset_parameters(self):
        for layer in self.residual:
            layer.reset_parameters()
        for layer in self.nonresidual:
            layer.reset_parameters()


class DPCNN(AEmbeddingModel):

    def __init__(self, embeddings_name, blocks, fc_layers, pool,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False,
                 char=False):
        super(DPCNN, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size,
                                    num_features=num_features, numpy_embeddings=numpy_embeddings)

        self.blocks = nn.Sequential(*[ResidualBlock(**block) for block in blocks])
        self.fc_layers = nn.Sequential(*[layers.FullyConnected(**fc_layer) for fc_layer in fc_layers])
        self.pool = build_pyjet_layer(**pool)
        self.char = char
        self.min_len = 5
        self.min_input_size = self.calc_input_size(self.min_len)

    def calc_input_size(self, output_size):
        for block in self.blocks:
            output_size = block.calc_input_size(output_size)
        return output_size

    def cast_input_to_torch(self, x, volatile=False):
        if self.char:
            x = [np.array(sample[:1024]) for sample in x]

        # Remove any missing words
        else:
            x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # Get the maximum length of the batch to not be degenerate
        # max_len = max(len(seq) for seq in x)
        # max_len = max(max_len, self.calc_input_size(self.min_len))
        x = [L.pad_numpy_to_length(sample, length=self.min_input_size) for sample in x]
        # return self.embeddings(Variable(J.from_numpy(x).long(), volatile=volatile))

        return [self.embeddings(Variable(J.from_numpy(sample).long(), volatile=volatile)) for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        x = self.blocks(x)
        # print(len(x), x[0].size())
        x = self.pool(x)
        # print(x.size())
        x = L.flatten(x)  # B x F*k
        # print(x.size())
        # Run the fc layer if we have one
        x = self.fc_layers(x)
        self.loss_in = x
        return self.loss_in

    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()
        self.pool.reset_parameters()
        for layer in self.fc_layers:
            layer.reset_parameters()


registry.register_model("cnn-emb", CNNEmb)
registry.register_model("dpcnn", DPCNN)