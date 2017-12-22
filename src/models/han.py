import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from pyjet.models import SLModel
import pyjet.backend as J

from models.modules import pad_torch_embedded_sequences, unpad_torch_embedded_sequences, pack_torch_embedded_sequences, \
    unpack_torch_embedded_sequences, pad_numpy_to_length


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
        self.dropout = nn.Dropout(encoder_dropout)
        self.dropout_prob = encoder_dropout

    def forward(self, x):
        raise NotImplementedError()


class GRUEncoder(Encoder):

    def __init__(self, input_encoding_size, output_encoding_size, encoder_dropout=0.2, batchnorm=False):
        super(GRUEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        assert self.output_encoding_size % 2 == 0
        self.encoder = nn.GRU(input_encoding_size,
                              output_encoding_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        # self.batchnorm = batchnorm
        # Removed because this seems to hamper performance
        # self.bn = nn.BatchNorm1d(self.output_encoding_size) if self.batchnorm else None

    def forward(self, x):
        # x input is B x Li x I
        x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
        x, _ = self.encoder(x)  # B x L x H
        # if self.batchnorm:
        #     x = self.bn(x.transpose(1, 2).contiguous()).transpose(1, 2)
        x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
        if self.dropout_prob != 1:
            x = [self.dropout(seq) for seq in x]
        return x


class LSTMEncoder(Encoder):

    def __init__(self, input_encoding_size, output_encoding_size, encoder_dropout=0.2, batchnorm=False):
        super(LSTMEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        assert self.output_encoding_size % 2 == 0
        self.encoder = nn.LSTM(input_encoding_size,
                               output_encoding_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        # self.batchnorm = batchnorm
        # Removed because this seems to hamper performance
        # self.bn = nn.BatchNorm1d(self.output_encoding_size) if self.batchnorm else None

    def forward(self, x):
        # x input is B x Li x I
        x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
        x, _ = self.encoder(x)  # B x L x H
        # if self.batchnorm:
        #     x = self.bn(x.transpose(1, 2).contiguous()).transpose(1, 2)
        x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
        if self.dropout_prob != 1:
            x = [self.dropout(seq) for seq in x]
        return x


class ConvEncoder(Encoder):

    def __init__(self, input_encoding_size, output_encoding_size, kernel_size, encoder_dropout=0.2, batchnorm=False):
        super(ConvEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        self.padding = (kernel_size - 1) // 2
        self.encoder = nn.Conv1d(input_encoding_size, output_encoding_size, kernel_size, padding=self.padding)
        # Removed because this seems to hamper performance
        # self.batchnorm = batchnorm
        # self.bn = nn.BatchNorm1d(self.output_encoding_size) if self.batchnorm else None

    def forward(self, x):
        # x input is B x Li x I
        x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
        x = F.relu(self.encoder(x.transpose(1, 2).contiguous()))  # B x H x L
        # if self.batchnorm:
        #     x = self.bn(x)
        x = x.transpose(1, 2)  # B x L x H
        x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
        if self.dropout_prob != 1:
            x = [self.dropout(seq) for seq in x]
        return x


class AttentionHierarchy(nn.Module):

    def __init__(self, input_encoding_size, hidden_size, encoder_dropout=0.2, batchnorm=False):
        super(AttentionHierarchy, self).__init__()

        self.input_encoding_size = input_encoding_size
        self.hidden_size = hidden_size
        # We want to swap this out with other encoders
        # self.encoder = ConvEncoder(input_encoding_size, hidden_size, 3, encoder_dropout=encoder_dropout,
        #                            batchnorm=batchnorm)
        # self.encoder2 = ConvEncoder(hidden_size, hidden_size, 3, encoder_dropout=encoder_dropout,
        #                             batchnorm=batchnorm)
        self.encoder = GRUEncoder(input_encoding_size, hidden_size, encoder_dropout=encoder_dropout, batchnorm=batchnorm)
        # self.encoder = LSTMEncoder(input_encoding_size, hidden_size,1 encoder_dropout=encoder_dropout,
        # batchnorm=batchnorm)
        self.att = AttentionBlock(hidden_size, hidden_size, batchnorm=batchnorm)
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.hidden_size)

    def forward(self, x):
        # x input is B x Li x I
        x = self.encoder(x)  # B x Li x H
        # x = self.encoder2(x)
        x = self.att(x)  # B x Li x H
        # Sum up each sequence
        return torch.stack([seq.sum(dim=0) for seq in x])  # B x H


class HAN(SLModel):

    def __init__(self, n_features, n_hidden_word=100, n_hidden_sent=100, encoder_dropout=0.2, fc1_size=0,
                 fc_dropout=0.5, batchnorm=False, logger=None):
        super(HAN, self).__init__()

        self.n_features = n_features
        self.n_hidden_word = n_hidden_word
        self.n_hidden_sent = n_hidden_sent
        self.word_hierarchy = AttentionHierarchy(n_features, n_hidden_word, encoder_dropout=encoder_dropout,
                                                 batchnorm=batchnorm)
        self.sent_hierarchy = AttentionHierarchy(n_hidden_word, n_hidden_sent, encoder_dropout=encoder_dropout,
                                                 batchnorm=batchnorm)
        self.batchnorm = batchnorm
        # Fully connected layers
        self.has_fc = fc1_size > 0
        if self.has_fc:
            self.dropout_fc1 = nn.Dropout(fc_dropout) if fc_dropout != 1. else lambda x: x
            self.bn_fc = nn.BatchNorm1d(fc1_size)
            self.fc1 = nn.Linear(n_hidden_sent, fc1_size)
            self.fc_eval = nn.Linear(fc1_size, 6)
        else:
            self.fc1 = None
            self.fc_eval = nn.Linear(n_hidden_sent, 6)
        self.dropout_fc_eval = nn.Dropout(fc_dropout) if fc_dropout != 1. else lambda x: x

    def cast_input_to_torch(self, x, volatile=False):
        # x comes in as a batch of list of embedded sentences
        # Need to turn it into a list of packed sequences
        # We make packs for each document
        x = [[pad_numpy_to_length(sent, length=1) for sent in sample] for sample in x]
        x = [(sample if len(sample) > 0 else [pad_numpy_to_length(np.empty((0, self.n_features)), 1)]) for sample in x]
        return [[Variable(J.Tensor(sent), volatile) for sent in sample] for sample in x]

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
            sent_outs = self.dropout_fc1(sent_outs)
            sent_outs = self.fc1(sent_outs)
            sent_outs = self.bn_fc(sent_outs)

        sent_outs = self.dropout_fc_eval(sent_outs)
        self.loss_in = self.fc_eval(sent_outs)  # B x 6
        return F.sigmoid(self.loss_in)

