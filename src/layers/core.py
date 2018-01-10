import torch
import torch.nn as nn
import torch.nn.functional as F
import layers.functions as L
import layers.utils as utils
import logging
import warnings


class RNN(nn.Module):
    def __init__(self, input_size, output_size, rnn_type='gru', n_layers=1, input_dropout=0.0, dropout=0.0,
                 bidirectional=False, residual=False):
        super(RNN, self).__init__()
        output_size = output_size // 2 if bidirectional else output_size
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, output_size, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional,
                              batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, output_size, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional,
                               batch_first=True)
        self.input_dropout_p = input_dropout
        self.residual = residual
        # Logging
        logging.info("Created a{} {} cell with {} input neurons and {} output neurons".format(
            " bidirectional" if bidirectional else "", rnn_type, input_size, output_size))
        logging.info("Using {} layers, {} rnn dropout, {} input dropout".format(n_layers, dropout, input_dropout))
        logging.info("Using {} residual connection".format("a" if residual else "no"))

    def forward(self, x):
        if self.input_dropout_p:
            x = [F.dropout(sample.unsqueeze(0), p=self.input_dropout_p, training=self.training).squeeze(0) for sample in
                 x]
        x, seq_lens = L.pad_torch_embedded_sequences(x)  # B x L x I
        if self.residual:
            residual = x
        x, _ = self.rnn(x)
        if self.residual:
            x = residual + x
        x = L.unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x H
        return x


class Conv1d(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, dilation=1, activation='linear', padding='same',
                 pool_type='no_pool', pool_padding='same', pool_kernel_size=2, pool_stride=1, pool_dilation=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv1d, self).__init__()
        if padding != 'same' and not isinstance(padding, int):
            raise NotImplementedError("padding: %s" % padding)
        if pool_padding != 'same' and not isinstance(pool_padding, int):
            raise NotImplementedError("pool padding: %s" % pool_padding)
        padding = (kernel_size - 1) // 2 if padding == 'same' else padding
        pool_padding = (pool_kernel_size - 1) // 2 if pool_padding == 'same' else pool_padding

        # Set these as attributes to calculate the output size
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pool_padding = pool_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_dilation = pool_dilation
        self.pool_name = pool_type

        self.conv = nn.Conv1d(input_size, output_size, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.activation = utils.get_activation_type(activation)
        self.pool_type = utils.get_pool_type(pool_type)
        self.pool = self.pool_type(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding,
                                   dilation=pool_dilation)
        self.bn = nn.BatchNorm1d(output_size) if batchnorm else None
        self.input_dropout_p = input_dropout
        self.dropout_p = dropout

        # Logging
        logging.info("Using Conv1d layer with {} input, {} output, {} kernel, {} stride, and {} padding".format(
            input_size, output_size, kernel_size, stride, padding))
        logging.info("Using activation %s" % self.activation.__name__)
        if self.pool_name != "no_pool":
            logging.info("Using pool {} with {} kernel size, {} stride, and {} padding".format(
                self.pool_type.__name__, self.pool_kernel_size, self.pool_stride, self.pool_padding))
        else:
            logging.info("Not using any pool.")
        logging.info("Using batchnorm1d" if self.bn is not None else "Not using batchnorm1d")
        logging.info("Using {} input dropout and {} dropout".format(self.input_dropout_p, self.dropout_p))

    def calc_output_size(self, input_size):
        output_size = (input_size - self.dilation * (self.kernel_size - 1) + 2 * self.padding - 1) // self.stride + 1
        if self.pool_name != "no_pool":
            output_size = (output_size - self.pool_dilation * (
                self.pool_kernel_size - 1) + 2 * self.pool_padding - 1) // self.pool_stride + 1
        return output_size

    def forward(self, x):
        # Change the length dimension for the convolution operations
        x = [seq.transpose(0, 1) for seq in x]
        if self.input_dropout_p:
            x = [F.dropout(sample.unsqueeze(0), p=self.input_dropout_p, training=self.training).squeeze(0) for sample in
                 x]
        x = [self.pool(self.activation(self.conv(sample.unsqueeze(0)))).squeeze(0) for sample in x]  # B x F x Li-P
        # Do the batchnorm
        if self.bn is not None:
            x, lens = L.pad_torch_embedded_sequences(x, length_last=True)
            x = self.bn(x)
            x = L.unpad_torch_embedded_sequences(x, lens, length_last=True)
        if self.dropout_p:
            x = [F.dropout(sample.unsqueeze(0), p=self.dropout_p, training=self.training).squeeze(0) for sample in x]
        # Undo the length dimension change for the convolution operations
        x = [seq.transpose(0, 1) for seq in x]
        return x


class FullyConnected(nn.Module):

    def __init__(self, input_size, output_size, bias=True, activation='linear',
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(FullyConnected, self).__init__()
        self.n_input = input_size
        self.n_output = output_size
        self.activation_name = activation

        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.activation = utils.get_activation_type(activation)
        self.bn = nn.BatchNorm1d(output_size) if batchnorm else None
        self.input_dropout_p = input_dropout
        self.dropout_p = dropout
        # Logging
        logging.info("Using Linear layer with {} input, {} output, and {}".format(
            input_size, output_size, "bias" if bias else "no bias"))
        logging.info("Using activation %s" % self.activation.__name__)
        logging.info("Using batchnorm1d" if self.bn is not None else "Not using batchnorm1d")
        logging.info("Using {} input dropout and {} dropout".format(self.input_dropout_p, self.dropout_p))

    def forward(self, x):
        if self.input_dropout_p:
            x = F.dropout(x, p=self.input_dropout_p, training=self.training)
        x = self.activation(self.linear(x))
        if self.bn is not None:
            x = self.bn(x.unsqueeze(-1)).squeeze(-1)
        if self.dropout_p:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def __str__(self):
        return "{} FullyConnected {} x {}".format(self.activation_name, self.n_input, self.n_output)


class TimeDistributed(nn.Module):

    def __init__(self, layer, batchnorm=False):
        super(TimeDistributed, self).__init__()
        self.layer = layer
        self.bn = nn.BatchNorm1d(layer.n_output) if batchnorm else None
        logging.info("TimeDistributiong %s layer" % self.layer)
        if batchnorm:
            logging.info("Using batchnorm")

    def forward(self, x):
        x, seq_lens = L.pack_torch_embedded_sequences(x)  # B*Li x I
        x = self.layer(x)  # B*Li x O
        x = L.unpack_torch_embedded_sequences(x, seq_lens)
        if self.bn is not None:
            x, seq_lens = L.pad_torch_embedded_sequences(x)  # B x L x O
            x = x.transpose(1, 2)  # B x O x L
            x = self.bn(x.contiguous())
            x = x.transpose(1, 2)  # B x L x O
            x = L.unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x O
        return x


class TimeDistributedLinear(nn.Module):

    def __init__(self, input_size, output_size, num_weights=1, batchnorm=False, bias=True):
        super(TimeDistributedLinear, self).__init__()
        warnings.warn("Use TimeDistributed around a linear layer instead", DeprecationWarning)
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
        self.bn = nn.BatchNorm1d(self.output_size) if self.batchnorm else None
        # Logging
        logging.info("Using TimeDistributedLinear layer with {} input, {} output, {} weight sets, and {}".format(
            input_size, output_size, num_weights, "bias" if bias else "no bias"))
        logging.info("Using batchnorm1d" if self.bn is not None else "Not using batchnorm1d")

    def forward(self, x, nonlinearity=lambda x: x):
        # input comes in as B x Li x I
        # First we pack the data to run the linear over batch and length
        x, seq_lens = L.pack_torch_embedded_sequences(x)  # B*Li x I
        x = nonlinearity(self.weights(x))  # B*Li x O
        # unpack the data
        x = L.unpack_torch_embedded_sequences(x, seq_lens)
        if self.batchnorm:
            # pad the packed output
            x, seq_lens = L.pad_torch_embedded_sequences(x)  # B x L x O
            # Transpose the L and O so the input_size becomes the channels
            x = x.transpose(1, 2)  # B x O x L
            x = self.bn(x.contiguous())
            # Transpose the L and O so we can unpad the padded sequence
            x = x.transpose(1, 2)  # B x L x O

            x = L.unpad_torch_embedded_sequences(x, seq_lens)  # B x Li x O
        return x

