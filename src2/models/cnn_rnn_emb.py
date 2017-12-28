import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pyjet.backend as J

from models.abstract_model import AEmbeddingModel
from models.han import AttentionHierarchy
from models.modules import pad_torch_embedded_sequences, unpad_torch_embedded_sequences, pad_numpy_to_length


class CNNRNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_path, trainable=False, vocab_size=None, num_features=None, input_dropout=0.25,
                 kernel_size=3, pool_kernel_size=2, pool_stride=1, n1_filters=512, n2_filters=256, conv_dropout=0.25,
                 rnn_type='gru', rnn_size=256, num_layers=1, rnn_dropout=0.25,
                 fc_size=256, fc_dropout=0.25,
                 batchnorm=True):
        super(CNNRNNEmb, self).__init__(embeddings_path, trainable=trainable, vocab_size=vocab_size, num_features=num_features)

        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.rnn_dropout = rnn_dropout
        self.fc_dropout = fc_dropout
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.n1_filters = n1_filters
        self.n2_filters = n2_filters
        self.conv_dropout = conv_dropout
        self.padding = (kernel_size - 1) // 2

        # Dropout on embeddings
        self.embeddings_dropout = nn.Dropout(input_dropout) if input_dropout != 1. else lambda x: x

        # Block 1
        self.conv1 = nn.Conv1d(self.num_features, n1_filters, kernel_size, padding=self.padding)
        self.m1 = nn.MaxPool1d(pool_kernel_size, stride=pool_stride)
        self.bn1 = nn.BatchNorm1d(n1_filters) if batchnorm else None
        self.dropout1 = nn.Dropout(conv_dropout) if conv_dropout != 1. else lambda x: x
        # Block 2
        self.conv2 = nn.Conv1d(n1_filters, n2_filters, kernel_size, padding=self.padding)
        self.m2 = nn.MaxPool1d(pool_kernel_size, stride=pool_stride)
        self.bn2 = nn.BatchNorm1d(n2_filters) if batchnorm else None
        self.dropout2 = nn.Dropout(conv_dropout) if conv_dropout != 1. else lambda x: x

        # RNN Block
        self.rnn_module = AttentionHierarchy(self.n2_filters, rnn_size, num_layers=num_layers, encoder_type=rnn_type,
                                             encoder_dropout=rnn_dropout, batchnorm=batchnorm)

        # Fully connected layers
        self.has_fc = fc_size > 0
        if self.has_fc:
            self.dropout_fc = nn.Dropout(fc_dropout) if fc_dropout != 1. else lambda x: x
            self.bn_fc = nn.BatchNorm1d(fc_size)
            print("Creating fully connected layer of size %s" % fc_size)
            self.fc_layer = nn.Linear(rnn_size, fc_size)
            self.fc_eval = nn.Linear(fc_size, 6)
        else:
            print("Not creating fully connected layer")
            self.fc_layer = None
            self.fc_eval = nn.Linear(rnn_size, 6)
        print("Creating dropout with %s drop prob" % fc_dropout)
        self.dropout_fc_eval = nn.Dropout(fc_dropout) if fc_dropout != 1. else lambda x: x
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
        x = [self.embeddings_dropout(sample) for sample in x]  # B x Li x E
        # Pass through the conv module
        x = [sample.transpose(0, 1) for sample in x]
        x = [self.m1(F.relu(self.conv1(sample.unsqueeze(0)))).squeeze(0) for sample in x]  # B x 512 x Li-1
        # Do the batchnorm
        if self.batchnorm:
            x, lens = pad_torch_embedded_sequences(x, length_last=True)
            x = self.bn1(x)
            x = unpad_torch_embedded_sequences(x, lens, length_last=True)

        x = [self.dropout1(sample.unsqueeze(0)).squeeze(0) for sample in x]

        x = [self.m2(F.relu(self.conv2(sample.unsqueeze(0)))).squeeze(0) for sample in x]  # B x 256 x Li-2
        # Do the batchnorm
        if self.batchnorm:
            x, lens = pad_torch_embedded_sequences(x, length_last=True)
            x = self.bn2(x)
            x = unpad_torch_embedded_sequences(x, lens, length_last=True)

        x = [self.dropout2(sample.unsqueeze(0)).squeeze(0) for sample in x]
        x = [sample.transpose(0, 1) for sample in x]

        x = self.rnn_module(x)
        x = J.flatten(x)  # B x 256

        # Run the fc layer if we have one
        if self.has_fc:
            x = self.dropout_fc(x)
            x = F.relu(self.fc_layer(x))
            x = self.bn_fc(x)

        x = self.dropout_fc_eval(x)
        self.loss_in = self.fc_eval(x)  # B x 6
        return F.sigmoid(self.loss_in)
