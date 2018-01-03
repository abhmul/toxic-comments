import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyjet.backend as J
from models.abstract_model import AEmbeddingModel
from layers import build_layer, FullyConnected
import layers.functions as L
from registry import registry


# noinspection PyCallingNonCallable
# class FeatureWiseAttentionBlock(nn.Module):
#
#     def __init__(self, encoding_size, hidden_size, num_heads=1, batchnorm=False, att_type='tanh'):
#         super(FeatureWiseAttentionBlock, self).__init__()
#
#         self.encoding_size = encoding_size
#         self.hidden_size = hidden_size
#         self.batchnorm = batchnorm
#         if att_type == 'tanh':
#             self.att_nonlinearity = F.tanh
#             print("Using tanh for att nonlin")
#             self.hidden_layer = TimeDistributedLinear(self.encoding_size, self.hidden_size, batchnorm=batchnorm)
#             self.context_vector = TimeDistributedLinear(self.hidden_size, self.encoding_size, bias=False)
#         elif att_type =='relu':
#             self.att_nonlinearity = F.relu
#             print("Using relu for att nonlin")
#             self.hidden_layer = TimeDistributedLinear(self.encoding_size, self.hidden_size, batchnorm=batchnorm)
#             self.context_vector = TimeDistributedLinear(self.hidden_size, self.encoding_size, bias=False)
#         elif att_type == 'drelu':
#             self.att_nonlinearity = lambda x_list: F.relu(x_list[0]) - F.relu(x_list[1])
#             print("Using drelu for att nonlin")
#             self.hidden_layer = TimeDistributedLinear(self.encoding_size, self.hidden_size, num_weights=2, batchnorm=batchnorm)
#             self.context_vector = TimeDistributedLinear(self.hidden_size, self.encoding_size, bias=False)
#         elif att_type =='linear':
#             self.att_nonlinearity = lambda x: x
#             print("Not using any att nonlin")
#             # Just use a hidden layer and no context vector
#             self.hidden_layer = TimeDistributedLinear(self.encoding_size, self.encoding_size, batchnorm=batchnorm)
#             self.context_vector = lambda x: x
#         else:
#             raise NotImplementedError()
#
#     def forward(self, encoding_pack):
#         # The input comes in as B x Li x E
#         x = self.hidden_layer(encoding_pack, nonlinearity=self.att_nonlinearity)  # B x Li x H
#         att = self.context_vector(x)  # B x Li x E
#         att = timedistributed_softmax(att)  # B x Li x E
#         # Apply the attention
#         encoding_pack = [att_i * sample for att_i, sample in zip(att, encoding_pack)]
#         return torch.stack([sample.sum(0) for sample in encoding_pack]).unsqueeze(1)  # B x E


# class QRNNEncoder(Encoder):
#
#     def __init__(self, input_encoding_size, output_encoding_size, num_layers=1, encoder_dropout=0.2, dense=False):
#         super(QRNNEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)
#
#         if dense:
#             raise NotImplementedError("Dense")
#         assert self.output_encoding_size % 2 == 0
#         print("Creating QRNN cell with %s layers" % num_layers)
#         self.encoder = QRNN(input_encoding_size, output_encoding_size // 2, num_layers=num_layers,
#                             bidirectional=True, dropout=encoder_dropout, window=2)
#
#     def forward(self, x):
#         # x input is B x Li x I
#         x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
#         x = x.transpose(0, 1)
#         x, _ = self.encoder(x)  # L x B x H
#         x = x.transpose(0, 1)  # B x L x H
#         x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
#         # if self.dropout_prob != 1:
#         #     x = [self.dropout(seq.unsqueeze(0)).squeeze(0) for seq in x]  # B x Li x H
#         return x


class HAN(AEmbeddingModel):

    def __init__(self, embeddings_path, word_layers, word_pool, sent_layers, sent_pool, fc_layers,
                 trainable=False, vocab_size=None, num_features=None, numpy_embeddings=False):
        super(HAN, self).__init__(embeddings_path, trainable=trainable, vocab_size=vocab_size,
                                  num_features=num_features, numpy_embeddings=numpy_embeddings)

        self.word_layers = nn.ModuleList([build_layer(**word_layer) for word_layer in word_layers])
        self.word_pool = build_layer(**word_pool)
        self.sent_layers = nn.ModuleList([build_layer(**sent_layer) for sent_layer in sent_layers])
        self.sent_pool = build_layer(**sent_pool)
        self.fc_layers = nn.ModuleList([FullyConnected(**fc_layer) for fc_layer in fc_layers])

        self.min_len = 1
        self.default_sentence = np.zeros((self.min_len, self.num_features))

    def cast_input_to_torch(self, x, volatile=False):
        # x comes in as a batch of list of token sentences
        # Need to turn it into a list of packed sequences
        # We make packs for each document
        # Remove any missing words
        x = [[np.array([word for word in sent if word not in self.missing]) for sent in sample] for sample in x]
        x = [[L.pad_numpy_to_length(sent, length=self.min_len) for sent in sample] for sample in x]
        x = [(sample if len(sample) > 0 else [self.default_sentence]) for sample in x]
        return [[self.embeddings(Variable(J.from_numpy(sent).long(), volatile=False)) for sent in sample] for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(torch.from_numpy(y).cuda().float(), volatile=volatile)

    def forward(self, x):
        # x is B x Si x Wj x E
        # print([[tuple(sent.size()) for sent in sample] for sample in x])
        # We run on each sentence first
        # Each sample is Si x Wj x E
        word_outs = x
        for word_layer in self.word_layers:
            word_outs = [word_layer(sample) for sample in word_outs]
        word_outs = [L.flatten(self.word_pool(sample)) for sample in word_outs]
        # print([tuple(sent.size()) for sent in word_outs])
        # Run it on each sentence
        sent_outs = word_outs
        for sent_layer in self.sent_layers:
            sent_outs = sent_layer(sent_outs)
        sent_outs = L.flatten(self.sent_pool(sent_outs))
        # print(tuple(sent_outs.size()))
        # Run the fc layer if we have one
        x = sent_outs
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        self.loss_in = x  # B x 6
        return F.sigmoid(self.loss_in)


registry.register_model("han", HAN)
