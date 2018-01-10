import torch
import torch.nn as nn
import layers.functions as L
import layers.core as core
import layers.utils as utils
import logging
import warnings


class Pool1d(nn.Module):

    def __init__(self, pool_type, kernel_size, stride=None, padding='same', dilation=1):
        super(Pool1d, self).__init__()
        padding = (kernel_size - 1) // 2 if padding == 'same' else padding
        if dilation != 1:
            raise NotImplementedError("Dilation: %s" % dilation)
        pool_func = utils.get_pool_type(pool_type)
        self.pool = pool_func(kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        logging.info("Creating {} pooling layer with {} kernel, {} stride, and {} padding".format(pool_func.__name__,
                                                                                                  kernel_size, stride,
                                                                                                  padding))

    def calc_output_size(self, input_size):
        output_size = (input_size - self.dilation * (self.kernel_size - 1) + 2 * self.padding - 1) // self.stride + 1
        return output_size

    def forward(self, x):
        x = [seq.transpose(0, 1) for seq in x]
        x = [self.pool(seq.unsqueeze(0)).squeeze(0) for seq in x]
        return [seq.transpose(0, 1) for seq in x]


class MaxPool1d(Pool1d):

    def __init__(self, kernel_size, stride=None, padding='same', dilation=1):
        super(MaxPool1d, self).__init__("max", kernel_size, stride=stride, padding=padding, dilation=dilation)


class AvgPool1d(Pool1d):

    def __init__(self, kernel_size, stride=None, padding='same', dilation=1):
        super(AvgPool1d, self).__init__("avg", kernel_size, stride=stride, padding=padding, dilation=dilation)


class GlobalMaxPool1d(nn.Module):

    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        # The input comes in as B x Li x E
        return torch.stack([torch.max(seq, dim=0)[0] for seq in x])


class KMaxPool(nn.Module):

    def __init__(self, k):
        super(KMaxPool, self).__init__()
        self.k = k

    def forward(self, x):
        # B x Li x E
        return torch.stack([L.kmax_pooling(seq, 0, self.k) for seq in x])  # B x k x E


class Attention(nn.Module):

    def __init__(self, n_input, n_output=1):
        super(Attention, self).__init__()
        self.hidden_layer = core.TimeDistributed(core.FullyConnected(n_input, n_output), batchnorm=False)
        logging.info("Using Attention layer with {} input, {} output".format(n_input, n_output))

    def forward(self, x):
        # The input comes in as B x Li x E
        att = self.hidden_layer(x)  # B x Li x H
        att, _ = L.timedistributed_softmax(att, return_padded=True)  # B x L x K
        return torch.bmm(att.transpose(1, 2), L.pad_torch_embedded_sequences(x)[0])


class ContextAttention(nn.Module):

    def __init__(self, n_input, n_output=1, activation='tanh', batchnorm=False):
        super(ContextAttention, self).__init__()
        self.activation_name = activation
        self.hidden_layer = core.TimeDistributed(core.FullyConnected(n_input, n_input, activation=activation),
                                                 batchnorm=batchnorm)
        self.context_vector = core.TimeDistributed(core.FullyConnected(n_input, n_output, bias=False), batchnorm=False)

    def forward(self, x):
        # The input comes in as B x Li x E
        att = self.context_vector(self.hidden_layer(x))  # B x Li x H
        att, _ = L.timedistributed_softmax(att, return_padded=True)  # B x L x K
        return torch.bmm(att.transpose(1, 2), L.pad_torch_embedded_sequences(x)[0])


# noinspection PyCallingNonCallable
class AttentionBlock(nn.Module):

    def __init__(self, n_input, n_output=1, n_hidden=None, batchnorm=False, att_type='linear'):
        super(AttentionBlock, self).__init__()
        warnings.warn("AttentionBlock is deprecated, use another attentional pooling", DeprecationWarning)

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.batchnorm = batchnorm
        if att_type == 'tanh':
            self.att_nonlinearity = F.tanh
            logging.info("Using tanh for att nonlin")
            self.hidden_layer = core.TimeDistributedLinear(self.n_input, self.n_hidden, batchnorm=batchnorm)
            self.context_vector = core.TimeDistributedLinear(self.n_hidden, self.n_output, bias=False)
        elif att_type =='relu':
            self.att_nonlinearity = F.relu
            logging.info("Using relu for att nonlin")
            self.hidden_layer = core.TimeDistributedLinear(self.n_input, self.n_hidden, batchnorm=batchnorm)
            self.context_vector = core.TimeDistributedLinear(self.n_hidden, self.n_output, bias=False)
        elif att_type == 'drelu':
            self.att_nonlinearity = lambda x_list: F.relu(x_list[0]) - F.relu(x_list[1])
            logging.info("Using drelu for att nonlin")
            self.hidden_layer = core.TimeDistributedLinear(self.n_input, self.n_hidden, num_weights=2, batchnorm=batchnorm)
            self.context_vector = core.TimeDistributedLinear(self.n_hidden, self.n_output, bias=False)
        elif att_type == 'linear':
            self.att_nonlinearity = lambda x: x
            logging.info("Not using any att nonlin")
            # Just use a hidden layer and no context vector
            self.hidden_layer = core.TimeDistributedLinear(self.n_input, self.n_output, batchnorm=False)
            self.context_vector = lambda x: x
        else:
            raise NotImplementedError("att type:", att_type)
            # Logging
        logging.info("Using Attention Block layer with {} input, {} output, {} hidden".format(
            n_input, n_output, "no" if n_hidden is None else n_hidden))

    def forward(self, encoding_pack):
        # The input comes in as B x Li x E
        x = self.hidden_layer(encoding_pack, nonlinearity=self.att_nonlinearity)  # B x Li x H
        att = self.context_vector(x)  # B x Li x K
        att, _ = L.timedistributed_softmax(att, return_padded=True)  # B x L x K
        # bmm(B x K x L, B x L x E) = B x K x E
        return torch.bmm(att.transpose(1, 2), L.pad_torch_embedded_sequences(encoding_pack)[0])