import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J
from models.abstract_model import AEmbeddingModel
from models.han import AttentionHierarchy
from models.modules import pad_numpy_to_length
from registry import registry


class PredictionModule(nn.Module):

    def __init__(self, num_input, num_out, hiddens=tuple(), dropout=0.25, batchnorm=True):
        super(PredictionModule, self).__init__()

        # Fully connected layers
        hiddens = list(hiddens)
        layer_ins = [num_input] + hiddens
        layer_outs = hiddens + [num_out]
        self.layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(layer_ins, layer_outs)])
        print("Created layers:", " ".join(str((layer.in_features, layer.out_features)) for layer in self.layers))
        if batchnorm:
            print("Creating batchnorm layers")
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for hidden in hiddens])
        else:
            self.bns = [lambda x: x for _ in hiddens]
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in layer_ins])
        print("Created", len(self.dropouts), "dropout layer(s) with drop prob:", dropout)

    def forward(self, x):
        # Do all the hidden layers first
        for i in range(len(self.layers) - 1):
            x = self.dropouts[i](x)
            x = F.relu(self.layers[i](x))
            x = self.bns[i](x)
        # Now do the output layer
        x = self.dropouts[-1](x)
        return self.layers[-1](x)


class MHARNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_path, trainable=False, vocab_size=None, num_features=None, input_dropout=0.25, rnn_type='gru',
                 rnn_size=300, num_layers=1, rnn_dropout=0.25, fc_size=256, fc_dropout=0.25, batchnorm=True):
        super(MHARNNEmb, self).__init__(embeddings_path, trainable=trainable, vocab_size=vocab_size, num_features=num_features)

        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.rnn_dropout = rnn_dropout
        self.fc_dropout = fc_dropout
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        # Dropout on embeddings
        self.embeddings_dropout = nn.Dropout(input_dropout) if input_dropout != 1. else lambda x: x
        # RNN Block (6 heads for each output
        self.rnn_module = AttentionHierarchy(self.num_features, rnn_size, num_heads=6,
                                             num_layers=num_layers, encoder_type=rnn_type,
                                             encoder_dropout=rnn_dropout, batchnorm=batchnorm)

        # 6 Prediction layers
        hiddens = tuple() if not fc_size else (fc_size,)
        self.prediction_layers = nn.ModuleList(
            [PredictionModule(rnn_size, 1, hiddens=hiddens, dropout=fc_dropout, batchnorm=batchnorm) for _ in range(6)])
        self.min_len = 5

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # If a sample is too short extend it
        x = [pad_numpy_to_length(sample, length=self.min_len) for sample in x]
        # Transpose to get features x length
        return [self.embeddings(Variable(J.from_numpy(sample).long(), volatile=volatile)) for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        x = [self.embeddings_dropout(sample.unsqueeze(0)).squeeze(0) for sample in x]  # B x Li x E
        x = self.rnn_module(x)  # B x 6 x 300
        assert x.size(1) == 6
        # Run each head through its prediction layer
        self.loss_in = torch.cat([self.prediction_layers[i](x[:, i]) for i in range(6)], dim=1)  # B x 6
        return F.sigmoid(self.loss_in)


registry.register_model("mha-rnn-emb", MHARNNEmb)