from .core import *
from .functions import *
from .pooling import *
from .utils import *

import pyjet.layers as JLayers

layer_types = {
    # Core
    "fully-connected": FullyConnected,
    "rnn": RNN,
    "conv": Conv1d,
    # Pooling
    "maxpool": MaxPool1d,
    "avgpool": AvgPool1d,
    "global-maxpool": GlobalMaxPool1d,
    "k-maxpool": KMaxPool,
    "attention": Attention,
    "context-attention": ContextAttention
}

pyjet_layer_types = {
    # Core
    "fully-connected": JLayers.FullyConnected,
    "flatten": JLayers.Flatten,
    # Recurrent
    "simple-rnn": JLayers.SimpleRNN,
    "gru": JLayers.GRU,
    "lstm": JLayers.LSTM,
    # Convolutional
    "conv": JLayers.Conv1D,
    "seq-conv": JLayers.SequenceConv1D,
    # Pooling
    "maxpool": JLayers.MaxPooling1D,
    "seq-maxpool": JLayers.SequenceMaxPooling1D,
    "avgpool": JLayers.AveragePooling1D,
    "global-maxpool": JLayers.GlobalMaxPooling1D,
    "seq-global-maxpool": JLayers.SequenceGlobalMaxPooling1D,
    "seq-global-avgpool": JLayers.SequenceGlobalAveragePooling1D,
    "global-avgpool": JLayers.GlobalAveragePooling1D,
    "k-maxpool": JLayers.KMaxPooling1D,
    "context-att": JLayers.ContextAttention,
    "concat": JLayers.Concatenate,
    "none": lambda: JLayers.Identity
}


def get_layer_type(layer_type):
    return get_type(layer_type, layer_types, "layer type %s" % layer_type)


def get_pyjet_layer_type(layer_type):
    return get_type(layer_type, pyjet_layer_types, "pyjet layer type %s" % layer_type)


def build_pyjet_layer(name, **kwargs):
    return get_pyjet_layer_type(name)(**kwargs)


def build_layer(name, **kwargs):
    return get_layer_type(name)(**kwargs)
