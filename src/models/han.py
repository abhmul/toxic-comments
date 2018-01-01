import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchqrnn import QRNN

import pyjet.backend as J
from models.abstract_model import AEmbeddingModel
from models.modules import pad_torch_embedded_sequences, unpad_torch_embedded_sequences, pad_numpy_to_length, timedistributed_softmax
from models.layers import AttentionBlock, TimeDistributedLinear
from registry import registry


# noinspection PyCallingNonCallable
class FeatureWiseAttentionBlock(nn.Module):

    def __init__(self, encoding_size, hidden_size, num_heads=1, batchnorm=False, att_type='tanh'):
        super(FeatureWiseAttentionBlock, self).__init__()

        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm
        if att_type == 'tanh':
            self.att_nonlinearity = F.tanh
            print("Using tanh for att nonlin")
            self.hidden_layer = TimeDistributedLinear(self.encoding_size, self.hidden_size, batchnorm=batchnorm)
            self.context_vector = TimeDistributedLinear(self.hidden_size, self.encoding_size, bias=False)
        elif att_type =='relu':
            self.att_nonlinearity = F.relu
            print("Using relu for att nonlin")
            self.hidden_layer = TimeDistributedLinear(self.encoding_size, self.hidden_size, batchnorm=batchnorm)
            self.context_vector = TimeDistributedLinear(self.hidden_size, self.encoding_size, bias=False)
        elif att_type == 'drelu':
            self.att_nonlinearity = lambda x_list: F.relu(x_list[0]) - F.relu(x_list[1])
            print("Using drelu for att nonlin")
            self.hidden_layer = TimeDistributedLinear(self.encoding_size, self.hidden_size, num_weights=2, batchnorm=batchnorm)
            self.context_vector = TimeDistributedLinear(self.hidden_size, self.encoding_size, bias=False)
        elif att_type =='linear':
            self.att_nonlinearity = lambda x: x
            print("Not using any att nonlin")
            # Just use a hidden layer and no context vector
            self.hidden_layer = TimeDistributedLinear(self.encoding_size, self.encoding_size, batchnorm=batchnorm)
            self.context_vector = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, encoding_pack):
        # The input comes in as B x Li x E
        x = self.hidden_layer(encoding_pack, nonlinearity=self.att_nonlinearity)  # B x Li x H
        att = self.context_vector(x)  # B x Li x E
        att = timedistributed_softmax(att)  # B x Li x E
        # Apply the attention
        encoding_pack = [att_i * sample for att_i, sample in zip(att, encoding_pack)]
        return torch.stack([sample.sum(0) for sample in encoding_pack]).unsqueeze(1)  # B x E


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
                 batchnorm=False, dense=False):
        super(GRUEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        assert self.output_encoding_size % 2 == 0
        print("Creating GRU cell with %s layers" % num_layers)
        print("GRU encoder has %s hidden neurons, %s drop prob" % (output_encoding_size, encoder_dropout))
        self.dense = dense
        print("Training a dense rnn")
        input_sizes = [input_encoding_size] + [output_encoding_size]*(num_layers - 1)
        if dense:
            input_sizes = [sum(input_sizes[:i+1]) for i in range(len(input_sizes))]

        self.encoders = nn.ModuleList([nn.GRU(input_sizes[i], output_encoding_size // 2, num_layers=1, batch_first=True,
                                              bidirectional=True, dropout=encoder_dropout) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x):
        # x input is B x Li x I
        for layer in self.encoders:
            output, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
            output, _ = layer(output)
            output = unpad_torch_embedded_sequences(output, seq_lens)  # B x Li x H
            if self.dense:
                # print([(x_seq.size(), output_seq.size()) for x_seq, output_seq in zip(x, output)])
                x = [torch.cat([x_seq, output_seq], dim=1) for x_seq, output_seq in zip(x, output)]  # B x Li x I+H
            else:
                x = output   # B x Li x H
            if self.dropout_prob != 1:
                x = [self.dropout(seq.unsqueeze(0)).squeeze(0) for seq in x]  # B x Li x H
        return x


class LSTMEncoder(Encoder):

    def __init__(self, input_encoding_size, output_encoding_size, num_layers=1, encoder_dropout=0.2, dense=False):
        super(LSTMEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        if dense:
            raise NotImplementedError("Dense")
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
            x = [self.dropout(seq.unsqueeze(0)).squeeze(0) for seq in x]  # B x Li x H
        return x


class QRNNEncoder(Encoder):

    def __init__(self, input_encoding_size, output_encoding_size, num_layers=1, encoder_dropout=0.2, dense=False):
        super(QRNNEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        if dense:
            raise NotImplementedError("Dense")
        assert self.output_encoding_size % 2 == 0
        print("Creating QRNN cell with %s layers" % num_layers)
        self.encoder = QRNN(input_encoding_size, output_encoding_size // 2, num_layers=num_layers,
                            bidirectional=True, dropout=encoder_dropout, window=2)

    def forward(self, x):
        # x input is B x Li x I
        x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
        x = x.transpose(0, 1)
        x, _ = self.encoder(x)  # L x B x H
        x = x.transpose(0, 1)  # B x L x H
        x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
        # if self.dropout_prob != 1:
        #     x = [self.dropout(seq.unsqueeze(0)).squeeze(0) for seq in x]  # B x Li x H
        return x


class ConvEncoder(Encoder):

    def __init__(self, input_encoding_size, output_encoding_size, kernel_size, num_layers=1, encoder_dropout=0.2, dense=False):
        super(ConvEncoder, self).__init__(input_encoding_size, output_encoding_size, encoder_dropout)

        if dense:
            raise NotImplementedError("Dense")
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
            x = [self.dropout(seq.unsqueeze(0)).squeeze(0) for seq in x]  # B x Li x H
        return x


class AttentionHierarchy(nn.Module):

    def __init__(self, input_encoding_size, hidden_size, num_heads=1,
                 num_layers=1, encoder_dropout=0.2, encoder_type='gru', batchnorm=False, att_type='tanh',
                 block_type=None, dense=False):
        super(AttentionHierarchy, self).__init__()

        self.input_encoding_size = input_encoding_size
        self.hidden_size = hidden_size
        self.dense = dense
        # We want to swap this out with other encoders
        if encoder_type == 'gru':
            self.encoder = GRUEncoder(input_encoding_size, hidden_size, num_layers=num_layers, encoder_dropout=encoder_dropout, dense=dense)
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(input_encoding_size, hidden_size, num_layers=num_layers, encoder_dropout=encoder_dropout, dense=dense)
        elif encoder_type == 'conv':
            self.encoder = ConvEncoder(input_encoding_size, hidden_size, 3, num_layers=num_layers, encoder_dropout=encoder_dropout, dense=dense)
        elif encoder_type == 'qrnn':
            self.encoder = QRNNEncoder(input_encoding_size, hidden_size, num_layers=num_layers, encoder_dropout=encoder_dropout, dense=dense)
        else:
            raise NotImplementedError(encoder_type)

        if block_type == 'simple':
            # att_block = SimpleAttentionBlock
            raise NotImplementedError(block_type)
        elif block_type == 'featurewise':
            att_block = FeatureWiseAttentionBlock
        else:
            att_block = AttentionBlock

        output_size = input_encoding_size + hidden_size*num_layers if dense else hidden_size
        self.att = att_block(output_size, hidden_size, num_heads=num_heads, batchnorm=batchnorm, att_type=att_type)

    def forward(self, x):
        # x input is B x Li x I
        out = self.att(self.encoder(x))  # B x Li x H -> B x K x H
        return out


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
        word_outs = [J.flatten(self.word_hierarchy(sample)) for sample in x]  # B x Si x H
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


registry.register_model("han", HAN)
