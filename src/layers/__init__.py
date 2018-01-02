from .core import *
from .functions import *
from .pooling import *
from .utils import *

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


def get_layer_type(layer_type):
    return get_type(layer_type, layer_types, "layer type %s" % layer_type)


def build_layer(name, **kwargs):
    return get_layer_type(name)(**kwargs)
