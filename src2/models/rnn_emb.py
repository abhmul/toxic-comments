import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pyjet.backend as J

from models.abstract_model import AEmbeddingModel
from models.han import AttentionHierarchy
from models.modules import kmax_pooling, pad_torch_embedded_sequences, unpad_torch_embedded_sequences, pad_numpy_to_length


class RNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_path, trainable=False, vocab_size=None, num_features=None, rnn_type='lstm',
                 rnn_size=300, rnn_dropout=0.25, fc1_size=256, fc_dropout=0.25, batchnorm=True):
        super(RNNEmb, self).__init__(embeddings_path, trainable=trainable, vocab_size=vocab_size, num_features=num_features)

        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.rnn_dropout = rnn_dropout
        self.fc_dropout = fc_dropout
        self.batchnorm = batchnorm

        # RNN Block
        self.rnn_module = AttentionHierarchy(self.num_features, rnn_size, encoder_dropout=rnn_dropout,
                                             batchnorm=batchnorm)

        # Fully connected layers
        self.has_fc = fc1_size > 0
        if self.has_fc:
            self.dropout_fc1 = nn.Dropout(fc_dropout) if fc_dropout != 1. else lambda x: x
            self.bn_fc = nn.BatchNorm1d(fc1_size)
            self.fc_layer = nn.Linear(rnn_size, fc1_size)
            self.fc_eval = nn.Linear(fc1_size, 6)
        else:
            self.fc_layer = None
            self.fc_eval = nn.Linear(rnn_size, 6)
        self.dropout_fc_eval = nn.Dropout(fc_dropout) if fc_dropout != 1. else lambda x: x

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array(word for word in sample if word not in self.missing) for sample in x]
        # If a sample is too short extend it
        x = [pad_numpy_to_length(sample, length=(self.k + 2)) for sample in x]
        # Transpose to get features x length
        return [self.embeddings(Variable(J.from_numpy(sample), volatile=volatile)) for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        x = self.rnn_module(x)
        # Do the batchnorm
        if self.batchnorm:
            x, lens = pad_torch_embedded_sequences(x, length_last=True)
            x = self.bn1(x)
            x = unpad_torch_embedded_sequences(x, lens, length_last=True)

        x = J.flatten(x)  # B x 300

        # Run the fc layer if we have one
        if self.has_fc:
            x = self.dropout_fc1(x)
            x = F.relu(self.fc_layer(x))
            x = self.bn_fc(x)

        x = self.dropout_fc_eval(x)
        self.loss_in = self.fc_eval(x)  # B x 6
        return F.sigmoid(self.loss_in)
