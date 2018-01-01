import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J
from models.abstract_model import AEmbeddingModel
from models.modules import pad_numpy_to_length
from models.layers import RNNLayer, AttentionBlock, FullyConnectedLayer

from registry import registry


class RNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_name, rnn_layers, fc_layers, att_block,
                 trainable=False, vocab_size=None, num_features=None):
        super(RNNEmb, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size, num_features=num_features)

        # RNN Block
        self.rnn_layers = nn.ModuleList([RNNLayer(**rnn_layer) for rnn_layer in rnn_layers])
        self.att_block = AttentionBlock(**att_block)
        self.fc_layers = nn.ModuleList([FullyConnectedLayer(**fc_layer) for fc_layer in fc_layers])

        self.min_len = 1

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
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x) # B x Li x H
        x = self.att_block(x)
        x = J.flatten(x)  # B x k*H

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x  # B x 6
        return F.sigmoid(self.loss_in)


registry.register_model("rnn-emb", RNNEmb)
