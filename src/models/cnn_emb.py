import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import pyjet.backend as J
import pyjet.layers as layers

from models.abstract_model import AEmbeddingModel
from layers import build_pyjet_layer
import pyjet.layers.functions as L
from pyjet.layers import Conv1D, FullyConnected, MaskedLayer, Identity, MaskedInput
from registry import registry


class CNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_name, conv_layers, fc_layers, global_pool, pool, trainable=False, vocab_size=None, num_features=None):
        super(CNNEmb, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size, num_features=num_features)

        # CNN Block
        self.conv_layers = nn.ModuleList([MaskedLayer(Conv1D(**conv_layer)) for conv_layer in conv_layers])
        # Fully Connected Block
        self.fc_layers = nn.ModuleList([FullyConnected(**fc_layer) for fc_layer in fc_layers])
        self.pool = MaskedLayer(build_pyjet_layer(**pool), mask_value='min')
        self.global_pool = MaskedLayer(build_pyjet_layer(**global_pool), mask_value='min')

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # Get the seq lens and pad it
        seq_lens = J.LongTensor([max(len(sample), self.min_len) for sample in x])
        x = np.array([L.pad_numpy_to_length(sample, length=seq_lens.max()) for sample in x], dtype=int)
        return self.embeddings(Variable(J.from_numpy(x).long(), volatile=volatile)), seq_lens

    def forward(self, inputs):
        x, seq_lens = inputs
        # Do the conv layers
        for conv in self.conv_layers:
            x, seq_lens = conv(x, seq_lens)
            x, seq_lens = self.pool(x, seq_lens)

        # Do the global pooling
        x, seq_lens = self.global_pool(x, seq_lens)

        x = L.flatten(x)  # B x k*H
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x
        return self.loss_in

    def reset_parameters(self):
        for layer in self.conv_layers:
            layer.reset_parameters()
        self.global_pool.reset_parameters()
        self.pool.reset_parameters()
        for layer in self.fc_layers:
            layer.reset_parameters()


class ResidualBlock(nn.Module):

    def __init__(self, residual_layers, nonresidual_layers):
        super(ResidualBlock, self).__init__()
        self.residual = nn.ModuleList([])
        # Add the residual layers
        for i, layer in enumerate(residual_layers):
            self.residual.append(MaskedLayer(build_pyjet_layer(**layer),
                                             mask_value=("min" if "max" in layer["name"] else 0.0)))
        self.nonresidual = nn.ModuleList([])
        # Add the residual layers
        for i, layer in enumerate(nonresidual_layers):
            self.nonresidual.append(MaskedLayer(build_pyjet_layer(**layer),
                                                mask_value=("min" if "max" in layer["name"] else 0.0)))

        if len(self.residual) and self.residual[0].input_size != self.residual[-1].output_size:
            self.shortcut = MaskedLayer(Conv1D(self.residual[0].input_size, self.residual[-1].output_size, 1))
        else:
            self.shortcut = MaskedInput()

    def calc_input_size(self, output_size):
        for layer in reversed(self.residual):
            output_size = layer.calc_input_size(output_size)
        for layer in reversed(self.nonresidual):
            output_size = layer.calc_input_size(output_size)
        return output_size

    def forward(self, x, seq_lens):
        input_x = self.shortcut(x, seq_lens)
        # Do the residual layers
        for res in self.residual:
            x, seq_lens = res(x, seq_lens)

        # Apply the residual connection
        x = input_x + x

        # Do the nonresidual layers
        for nonres in self.nonresidual:
            x, seq_lens = nonres(x, seq_lens)

        return x, seq_lens

    def reset_parameters(self):
        for layer in self.residual:
            layer.reset_parameters()
        self.shortcut.reset_parameters()
        for layer in self.nonresidual:
            layer.reset_parameters()


class DPCNN(AEmbeddingModel):

    def __init__(self, embeddings_name, blocks, fc_layers, global_pool,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(DPCNN, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size,
                                    num_features=num_features, numpy_embeddings=numpy_embeddings)

        self.blocks = nn.ModuleList([ResidualBlock(**block) for block in blocks])
        self.fc_layers = nn.ModuleList([layers.FullyConnected(**fc_layer) for fc_layer in fc_layers])
        self.global_pool = MaskedLayer(build_pyjet_layer(**global_pool), mask_value='min')
        self.min_len = 5
        self.min_input_size = self.calc_input_size(self.min_len)

    def calc_input_size(self, output_size):
        for block in self.blocks:
            output_size = block.calc_input_size(output_size)
        return output_size

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # Get the seq lens and pad it
        seq_lens = J.LongTensor([max(len(sample), self.min_len) for sample in x])
        x = np.array([L.pad_numpy_to_length(sample, length=seq_lens.max()) for sample in x], dtype=int)
        return self.embeddings(Variable(J.from_numpy(x).long(), volatile=volatile)), seq_lens

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, inputs):
        x, seq_lens = inputs
        for block in self.blocks:
            x, seq_lens = block(x, seq_lens)

        x, _ = self.pool(x, seq_lens)

        x = L.flatten(x)  # B x F*k
        # Run the fc layer if we have one
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
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