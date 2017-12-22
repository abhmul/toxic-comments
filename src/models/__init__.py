from models.cnn_emb import CNNEmb
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


register_model("cnn-emb", CNNEmb)
cnn_emb_config = {'num_features': 300, 'kernel_size': 3, 'pool_kernel_size': 2, 'pool_stride': 1, 'n1_filters': 512,
                  'n2_filters': 256, 'k': 5, 'conv_dropout': 0.2, 'fc1_size': 100, 'fc_dropout': 0.5, 'batchnorm': True}
register_config("cnn-emb", cnn_emb_config)

register_model("han", HAN)
han_config = {'n_features': 300, 'n_hidden_word': 128, 'n_hidden_sent': 64, 'encoder_dropout': 0.2, 'fc1_size': 0,
              'fc_dropout': 0.5, 'batchnorm': True}
register_config("han", han_config)
