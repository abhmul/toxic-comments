from models.cnn_emb import CNNEmb
from models.rnn_emb import RNNEmb
from models.han import HAN

MODELS = {}
CONFIGS = {}


def register_model(model_name, model_class):
    MODELS[model_name] = model_class


def register_config(model_name, config):
    CONFIGS[model_name] = config


def load_model(model_name):
    return MODELS[model_name]


def load_config(model_name):
    return CONFIGS[model_name]


# CONFIGS
cnn_emb_config = {'kernel_size': 3, 'pool_kernel_size': 2, 'pool_stride': 1, 'n1_filters': 512,
                  'n2_filters': 256, 'k': 5, 'conv_dropout': 0.2, 'fc_size': 100, 'fc_dropout': 0.5, 'batchnorm': True}
# TODO
# - try tinkering with the dropout
#   - 0.5 on fc
#   - 0.5 on fc and rnn_dropout
# - try decreasing num params to 128-64
rnn_emb_config = {'rnn_type': 'lstm', 'rnn_size': 300, 'rnn_dropout': 0.25, 'fc_size': 256, 'fc_dropout': 0.5,
                  'batchnorm': True}
# Reasons this might not work as well?
# - changed dropout to be recurrent dropout, maybe change back? OR include the old dropout as well
# - maybe needs more parameters? Try doing 300 hidden_word, 256 hidden_sent
# - lstm might work better? Doubt this is an issue though, gru and lstm seem to have similar performance
han_config = {'n_hidden_word': 128, 'n_hidden_sent': 64, 'encoder_type': 'gru', 'encoder_dropout': 0.2, 'fc_size': 0,
              'fc_dropout': 0.5, 'batchnorm': True}

register_model("cnn-emb", CNNEmb)
register_config("cnn-emb", cnn_emb_config)

register_model("rnn-emb", RNNEmb)
register_config("rnn-emb", rnn_emb_config)

register_model("han", HAN)
register_config("han", han_config)
