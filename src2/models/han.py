import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import pyjet.backend as J

from models.modules import pad_torch_embedded_sequences, unpad_torch_embedded_sequences, pack_torch_embedded_sequences, \
    unpack_torch_embedded_sequences, pad_numpy_to_length
from models.abstract_model import AEmbeddingModel


def timedistributed_softmax(x):
    # x comes in as B x Li x F, we compute the softmax over Li for each F
    softmax = nn.Softmax2d()
    x, lens = pad_torch_embedded_sequences(x, pad_value=-float('inf'))  # B x L x F
    shape = tuple(x.size())
    assert len(shape) == 3
    x = softmax(x.unsqueeze(-1)).squeeze(-1)
    assert tuple(x.size()) == shape
    # Un-pad the tensor and return
    return unpad_torch_embedded_sequences(x, lens)  # B x Li x F


class TimeDistributedLinear(nn.Module):

    def __init__(self, input_size, output_size, batchnorm=False, bias=True):
        super(TimeDistributedLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batchnorm = batchnorm
        self.weights = nn.Linear(self.input_size, self.output_size, bias=bias)
        self.bn = nn.BatchNorm1d(
            self.output_size) if self.batchnorm else None

    def forward(self, x, nonlinearity=lambda x: x):
        # input comes in as B x Li x I
        # First we pack the data to run the linear over batch and length
        x, seq_lens = pack_torch_embedded_sequences(x)  # B*Li x I
        x = nonlinearity(self.weights(x))  # B*Li x O
        # unpack the data
        x = unpack_torch_embedded_sequences(x, seq_lens)
        if self.batchnorm:
            # pad the packed output
            x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x O
            # Transpose the L and O so the input_size becomes the channels
            x = x.transpose(1, 2)  # B x O x L
            x = self.bn(x.contiguous())
            # Transpose the L and O so we can unpad the padded sequence
            x = x.transpose(1, 2)  # B x L x O
            x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x O
        return x


# noinspection PyCallingNonCallable
class AttentionBlock(nn.Module):

    def __init__(self, encoding_size, hidden_size, batchnorm=False):
        super(AttentionBlock, self).__init__()

        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm
        self.hidden_layer = TimeDistributedLinear(
            self.encoding_size, self.hidden_size, batchnorm=batchnorm)
        self.context_vector = TimeDistributedLinear(
            self.hidden_size, 1, bias=False)

    def forward(self, encoding_pack):
        # The input comes in as B x Li x E
        x = self.hidden_layer(encoding_pack, nonlinearity=F.tanh)  # B x Li x H
        att = self.context_vector(x)  # B x Li x 1
        att = timedistributed_softmax(att)  # B x Li x 1
        # Apply the attention
        encoding_att_pack = [att_i * encoding_pack_i for att_i, encoding_pack_i in zip(att, encoding_pack)]
        return encoding_att_pack  # B x Li x E


class Encoder(nn.Module):
    def __init__(self, input_encoding_size, output_encoding_size, encoder_dropout=0.2):
        super(Encoder, self).__init__()

        self.input_encoding_size = input_encoding_size
        self.output_encoding_size = output_encoding_size
        self.dropout_prob = encoder_dropout
        self.dropout = nn.Dropout(encoder_dropout)

    def forward(self, x):
        raise NotImplementedError()


class GRUEncoder(Encoder):

    def __init__(self, input_encoding_size, output_encoding_size, num_layers=1, encoder_dropout=0.2,
                 batchnorm=False):
        super(GRUEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        assert self.output_encoding_size % 2 == 0
        print("Creating GRU cell with %s layers" % num_layers)
        print("GRU encoder has %s hidden neurons, %s drop prob" % (output_encoding_size, encoder_dropout))
        self.encoder = nn.GRU(input_encoding_size, output_encoding_size // 2, num_layers=num_layers, batch_first=True,
                              bidirectional=True, dropout=encoder_dropout)

    def forward(self, x):
        # x input is B x Li x I
        x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
        x, _ = self.encoder(x)  # B x L x H
        x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
        if self.dropout_prob != 1:
            x = [self.dropout(seq) for seq in x]
        return x


class LSTMEncoder(Encoder):

    def __init__(self, input_encoding_size, output_encoding_size, num_layers=1, encoder_dropout=0.2):
        super(LSTMEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        assert self.output_encoding_size % 2 == 0
        print("Creating LSTM cell with %s layers" % num_layers)
        self.encoder = nn.LSTM(input_encoding_size, output_encoding_size // 2, num_layers=num_layers, batch_first=True,
                               bidirectional=True, dropout=encoder_dropout)

    def forward(self, x):
        # x input is B x Li x I
        x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
        x, _ = self.encoder(x)  # B x L x H
        x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
        if self.dropout_prob != 1:
            x = [self.dropout(seq) for seq in x]
        return x


class ConvEncoder(Encoder):

    def __init__(self, input_encoding_size, output_encoding_size, kernel_size, num_layers=1, encoder_dropout=0.2):
        super(ConvEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        self.padding = (kernel_size - 1) // 2
        print("Creating Conv Encoder with %s layers" % num_layers)
        modules = [nn.Conv1d(input_encoding_size, output_encoding_size, kernel_size, padding=self.padding)] + \
                  [nn.Conv1d(output_encoding_size, output_encoding_size, kernel_size, padding=self.padding) for _ in
                   range(num_layers - 1)]
        self.encoders = nn.ModuleList(modules=modules)

    def forward(self, x):
        # x input is B x Li x I
        x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
        x = x.trainspose(1, 2).contiguous()  # B x I x L
        for encoder in self.encoders:
            x = F.relu(encoder(x))  # B x H x L
        x = x.transpose(1, 2)  # B x L x H
        x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
        if self.dropout_prob != 1:
            x = [self.dropout(seq) for seq in x]
        return x


class AttentionHierarchy(nn.Module):

    def __init__(self, input_encoding_size, hidden_size, num_layers=1, encoder_dropout=0.2, encoder_type='gru', batchnorm=False):
        super(AttentionHierarchy, self).__init__()

        self.input_encoding_size = input_encoding_size
        self.hidden_size = hidden_size
        # We want to swap this out with other encoders
        if encoder_type == 'gru':
            self.encoder = GRUEncoder(input_encoding_size, hidden_size, num_layers=num_layers, encoder_dropout=encoder_dropout)
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(input_encoding_size, hidden_size, num_layers=num_layers, encoder_dropout=encoder_dropout)
        elif encoder_type == 'conv':
            self.encoder = ConvEncoder(input_encoding_size, hidden_size, 3, num_layers=num_layers, encoder_dropout=encoder_dropout)

        self.att = AttentionBlock(hidden_size, hidden_size, batchnorm=batchnorm)

    def forward(self, x):
        # x input is B x Li x I
        x = self.encoder(x)  # B x Li x H
        x = self.att(x)  # B x Li x H
        # Sum up each sequence
        return torch.stack([seq.sum(dim=0) for seq in x])  # B x H


class HAN(AEmbeddingModel):

    def __init__(self, embeddings_path, trainable=False, vocab_size=None, num_features=None, n_hidden_word=100, n_hidden_sent=100, encoder_type='gru', encoder_dropout=0.2,
                 fc_size=0, fc_dropout=0.5, batchnorm=False):
        super(HAN, self).__init__(embeddings_path, trainable=trainable, vocab_size=vocab_size, num_features=num_features)

        self.n_hidden_word = n_hidden_word
        self.n_hidden_sent = n_hidden_sent
        self.word_hierarchy = AttentionHierarchy(self.num_features, n_hidden_word, encoder_dropout=encoder_dropout,
                                                 encoder_type=encoder_type, batchnorm=batchnorm)
        self.sent_hierarchy = AttentionHierarchy(n_hidden_word, n_hidden_sent, encoder_dropout=encoder_dropout,
                                                 encoder_type=encoder_type, batchnorm=batchnorm)
        self.batchnorm = batchnorm
        # Fully connected layers
        self.has_fc = fc_size > 0
        if self.has_fc:
            self.dropout_fc = nn.Dropout(fc_dropout) if fc_dropout != 1. else lambda x: x
            self.bn_fc = nn.BatchNorm1d(fc_size)
            self.fc1 = nn.Linear(n_hidden_sent, fc_size)
            self.fc_eval = nn.Linear(fc_size, 6)
        else:
            self.fc1 = None
            self.fc_eval = nn.Linear(n_hidden_sent, 6)
        self.dropout_fc_eval = nn.Dropout(fc_dropout) if fc_dropout != 1. else lambda x: x
        self.min_len = 5
        self.default_sentence = np.zeros((self.min_len, self.num_features))

    def cast_input_to_torch(self, x, volatile=False):
        # x comes in as a batch of list of token sentences
        # Need to turn it into a list of packed sequences
        # We make packs for each document
        # Remove any missing words
        x = [[np.array([word for word in sent if word not in self.missing]) for sent in sample] for sample in x]
        x = [[pad_numpy_to_length(sent, length=self.min_len) for sent in sample] for sample in x]
        x = [(sample if len(sample) > 0 else [self.default_sentence]) for sample in x]
        return [[self.embeddings(Variable(J.from_numpy(sent).long(), volatile=False)) for sent in sample] for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(torch.from_numpy(y).cuda().float(), volatile=volatile)

    def forward(self, x):
        # x is B x Si x Wj x E
        # print([[tuple(sent.size()) for sent in sample] for sample in x])
        # We run on each sentence first
        # Each sample is Si x Wj x E
        word_outs = [self.word_hierarchy(sample) for sample in x]  # B x Si x H
        # print([tuple(sent.size()) for sent in word_outs])
        # Run it on each sentence
        sent_outs = self.sent_hierarchy(word_outs)  # B x H
        # print(tuple(sent_outs.size()))
        # Run the fc layer if we have one
        if self.has_fc:
            sent_outs = self.dropout_fc(sent_outs)
            sent_outs = F.relu(self.fc1(sent_outs))
            sent_outs = self.bn_fc(sent_outs)

        sent_outs = self.dropout_fc_eval(sent_outs)
        self.loss_in = self.fc_eval(sent_outs)  # B x 6
        return F.sigmoid(self.loss_in)

