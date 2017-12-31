import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J

from models.abstract_model import AEmbeddingModel
from models.modules import kmax_pooling, pad_torch_embedded_sequences, unpad_torch_embedded_sequences, \
    pad_numpy_to_length
from models.layers import Conv1dLayer, FullyConnectedLayer
from registry import registry


class CNNEmb(AEmbeddingModel):

    def __init__(self, embeddings_name, conv_layers, fc_layers, k, trainable=False, vocab_size=None, num_features=None):
        super(CNNEmb, self).__init__(embeddings_name, trainable=trainable, vocab_size=vocab_size, num_features=num_features)

        self.conv_layers = nn.ModuleList([Conv1dLayer(**conv_layer) for conv_layer in conv_layers])
        self.fc_layers = nn.ModuleList([FullyConnectedLayer(**fc_layer) for fc_layer in fc_layers])
        self.k = k

    def cast_input_to_torch(self, x, volatile=False):
        # Remove any missing words
        x = [np.array([word for word in sample if word not in self.missing]) for sample in x]
        # If a sample is too short extend it
        x = [pad_numpy_to_length(sample, length=(self.k + 2)) for sample in x]
        # Transpose to get features x length
        return [self.embeddings(Variable(J.from_numpy(sample).long(), volatile=volatile)).transpose(0, 1) for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y).float(), volatile=volatile)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = torch.stack([kmax_pooling(sample.unsqueeze(0), 2, self.k).squeeze(0) for sample in x])  # B x F x k
        x = J.flatten(x)  # B x F*k

        # Run the fc layer if we have one
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x
        return F.sigmoid(self.loss_in)


registry.register_model("cnn-emb", CNNEmb)