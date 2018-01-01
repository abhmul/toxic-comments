import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import pad_torch_embedded_sequences, unpad_torch_embedded_sequences, \
    pack_torch_embedded_sequences, unpack_torch_embedded_sequences, timedistributed_softmax

pool_types = {"no_pool": lambda kernel_size, stride: lambda x: x, "max": nn.MaxPool1d}
activation_types = {"linear": lambda x: x, "relu": F.relu, "softmax": F.softmax, "tanh": F.tanh}


def get_type(item_type, type_dict, fail_message):
    try:
        return type_dict[item_type]
    except KeyError:
        raise NotImplementedError(fail_message)


def get_pool_type(pool_type):
    return get_type(pool_type, pool_types, "pool type %s" % pool_type)


def get_activation_type(activation_type):
    return get_type(activation_type, activation_types, "Activation %s" % activation_type)


class RNNLayer(nn.Module):
    def __init__(self, n_input, n_output, rnn_type='gru', n_layers=1, input_dropout=0.0, dropout=0.0, bidirectional=False):
        super(RNNLayer, self).__init__()
        n_output = n_output // 2 if bidirectional else n_output
        if rnn_type == 'gru':
            self.rnn = nn.GRU(n_input, n_output, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(n_input, n_output, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        self.input_dropout_p = input_dropout

    def forward(self, x):
        if self.input_dropout_p:
            x = [F.dropout(sample.unsqueeze(0), p=self.input_dropout_p, training=self.training).squeeze(0) for sample in
                 x]
        x, seq_lens = pad_torch_embedded_sequences(x)  # B x L x I
        x, _ = self.rnn(x)
        x = unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
        return x


class Conv1dLayer(nn.Module):
    def __init__(self, n_input, n_output, kernel_size=3, stride=1, activation='linear', padding='same',
                 pool_type='no_pool', pool_kernel_size=2, pool_stride=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv1dLayer, self).__init__()
        if padding != 'same' and not isinstance(padding, int):
            raise NotImplementedError("padding: %s" % padding)
        padding = (kernel_size - 1) // 2 if padding == 'same' else padding
        self.conv = nn.Conv1d(n_input, n_output, kernel_size, stride=stride, padding=padding)
        self.activation = get_activation_type(activation)
        self.pool = get_pool_type(pool_type)(kernel_size=pool_kernel_size, stride=pool_stride)
        self.bn = nn.BatchNorm1d(n_output) if batchnorm else None
        self.input_dropout_p = input_dropout
        self.dropout_p = dropout

    def forward(self, x):
        if self.input_dropout_p:
            x = [F.dropout(sample.unsqueeze(0), p=self.input_dropout_p, training=self.training).squeeze(0) for sample in
                 x]
        x = [self.pool(self.activation(self.conv(sample.unsqueeze(0)))).squeeze(0) for sample in x]  # B x F x Li-P
        # Do the batchnorm
        if self.bn is not None:
            x, lens = pad_torch_embedded_sequences(x, length_last=True)
            x = self.bn(x)
            x = unpad_torch_embedded_sequences(x, lens, length_last=True)
        if self.dropout_p:
            x = [F.dropout(sample.unsqueeze(0), p=self.dropout_p, training=self.training).squeeze(0) for sample in x]
        return x


class FullyConnectedLayer(nn.Module):

    def __init__(self, n_input, n_output, bias=True, activation='linear',
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(FullyConnectedLayer, self).__init__()
        self.linear = nn.Linear(n_input, n_output, bias=bias)
        self.activation = get_activation_type(activation)
        self.bn = nn.BatchNorm1d(n_output) if batchnorm else None
        self.input_dropout_p = input_dropout
        self.dropout_p = dropout

    def forward(self, x):
        if self.input_dropout_p:
            x = F.dropout(x, p=self.input_dropout_p, training=self.training)
        x = self.activation(self.linear(x))
        if self.bn is not None:
            x = self.bn(x.unsqueeze(-1)).squeeze(-1)
        if self.dropout_p:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class TimeDistributedLinear(nn.Module):

    def __init__(self, input_size, output_size, num_weights=1, batchnorm=False, bias=True):
        super(TimeDistributedLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batchnorm = batchnorm
        # We have to do this so we can maintain compatibility with older models
        if num_weights == 1:
            self.weights = nn.Linear(self.input_size, self.output_size, bias=bias)
        else:
            self.weights_list = nn.ModuleList(
                [nn.Linear(self.input_size, self.output_size, bias=bias) for _ in range(num_weights)])
            self.weights = lambda x: [w(x) for w in self.weights_list]
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

    def __init__(self, n_input, n_output=1, n_hidden=None, batchnorm=False, att_type='linear'):
        super(AttentionBlock, self).__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.batchnorm = batchnorm
        if att_type == 'tanh':
            self.att_nonlinearity = F.tanh
            print("Using tanh for att nonlin")
            self.hidden_layer = TimeDistributedLinear(self.n_input, self.n_hidden, batchnorm=batchnorm)
            self.context_vector = TimeDistributedLinear(self.n_hidden, self.n_output, bias=False)
        elif att_type =='relu':
            self.att_nonlinearity = F.relu
            print("Using relu for att nonlin")
            self.hidden_layer = TimeDistributedLinear(self.n_input, self.n_hidden, batchnorm=batchnorm)
            self.context_vector = TimeDistributedLinear(self.n_hidden, self.n_output, bias=False)
        elif att_type == 'drelu':
            self.att_nonlinearity = lambda x_list: F.relu(x_list[0]) - F.relu(x_list[1])
            print("Using drelu for att nonlin")
            self.hidden_layer = TimeDistributedLinear(self.n_input, self.n_hidden, num_weights=2, batchnorm=batchnorm)
            self.context_vector = TimeDistributedLinear(self.n_hidden, self.n_output, bias=False)
        elif att_type == 'linear':
            self.att_nonlinearity = lambda x: x
            print("Not using any att nonlin")
            # Just use a hidden layer and no context vector
            self.hidden_layer = TimeDistributedLinear(self.n_input, self.n_output, batchnorm=batchnorm)
            self.context_vector = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, encoding_pack):
        # The input comes in as B x Li x E
        x = self.hidden_layer(encoding_pack, nonlinearity=self.att_nonlinearity)  # B x Li x H
        att = self.context_vector(x)  # B x Li x K
        att, _ = timedistributed_softmax(att, return_padded=True)  # B x L x K
        # bmm(B x K x L, B x L x E) = B x K x E
        return torch.bmm(att.transpose(1, 2), pad_torch_embedded_sequences(encoding_pack)[0])


layer_types = {"conv1d": Conv1dLayer, "fully-connected": FullyConnectedLayer}


def get_layer_type(layer_type):
    return get_type(layer_type, layer_types, "Layer %s" % layer_type)