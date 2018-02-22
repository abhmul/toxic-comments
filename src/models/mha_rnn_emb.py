import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import pyjet.backend as J
from models.abstract_model import AEmbeddingModel
import pyjet.layers.functions as L
from pyjet.layers import RNN, FullyConnected, Conv1D, ContextAttention, ContextMaxPool1D
from registry import registry


class MHARNNEmb(AEmbeddingModel):

    pool_types = {"context-attention": ContextAttention, "context-maxpool": ContextMaxPool1D}

    def __init__(self, embeddings_name, rnn_layers, fc_layers, pool_type, resample=False,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(MHARNNEmb, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size,
                                     num_features=num_features, numpy_embeddings=numpy_embeddings)

        # Some legacy handling
        rnn_func = RNN
        fully_connected_func = FullyConnected

        # RNN Block
        self.rnn_layers = nn.ModuleList([rnn_func(**rnn_layer) for rnn_layer in rnn_layers])

        # Attentional Layers
        att_name = pool_type["name"]
        att_batchnorm = pool_type["batchnorm"]
        att_dropout = pool_type["dropout"]
        att_activation = pool_type["activation"]
        self.att = self.pool_types[att_name](input_size=rnn_layers[-1]["output_size"], output_size=6,
                                             batchnorm=att_batchnorm, dropout=att_dropout, activation=att_activation,
                                             padded_input=True)

        self.fc_layers = nn.ModuleList()
        # The toxic fc layer
        self.fc_layers.append(nn.Sequential(*(fully_connected_func(**fc_layer) for fc_layer in fc_layers)))
        # Replace the input size to be double
        fc_layers[0]["input_size"] *= 2
        for _ in range(1, 6):
            self.fc_layers.append(nn.Sequential(*(fully_connected_func(**fc_layer) for fc_layer in fc_layers)))

        self.resample = resample and self.num_features != self.rnn_layers[0].input_size
        if self.resample:
            self.resampler = Conv1D(self.num_features, self.rnn_layers[0].input_size, 1, use_bias=False)

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

        # Apply the attention
        x = self.att(x, seq_lens=seq_lens)  # B x 6 x H

        preds = [None for _ in range(6)]
        # Run the toxic output
        preds[0] = self.fc_layers[0](x[:, 0])  # B x 1
        # Run the other
        for i in range(1, 6):
            preds[i] = self.fc_layers[i](torch.cat([x[:, 0], x[:, i]], dim=-1))

        # Combine them
        x = torch.cat(preds, dim=-1)

        self.loss_in = x  # B x 6
        return self.loss_in

    def reset_parameters(self):
        for layer in self.rnn_layers:
            layer.reset_parameters()
        self.att.reset_parameters()
        for layers in self.fc_layers:
            for layer in layers:
                layer.reset_parameters()
        if self.resample:
            self.resampler.reset_parameters()


registry.register_model("mha-rnn-emb", MHARNNEmb)