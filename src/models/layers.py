import torch.nn as nn
import torch.nn.functional as F
from models.modules import pad_torch_embedded_sequences, unpad_torch_embedded_sequences

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


layer_types = {"conv1d": Conv1dLayer, "fully-connected": FullyConnectedLayer}


def get_layer_type(layer_type):
    return get_type(layer_type, layer_types, "Layer %s" % layer_type)