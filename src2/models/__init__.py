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
#   - 0.5 on fc - No noticeable advantage (score is slightly worse)
#   - 0.5 on fc and rnn_dropout - based on above, prolly note worth it
# - been using gru actually the whole time. Definitely better by a small amount
# - try training without the fully connected layer
# - try decreasing num params to 128-64
# - try making the rnn 2 layers
rnn_emb_config = {'rnn_type': 'gru', 'rnn_size': 300, 'rnn_dropout': 0.25, 'fc_size': 0, 'fc_dropout': 0.25,
                  'num_layers': 1, 'batchnorm': True}
# Reasons this might not work as well?
# SOLVED: Some kind of issue with the new process_data. Porting over the old data fixed it.
# TODO
# - try using the modified process_data now to process the texts and see if the problem was fixed
# - try using an lstm - gru is just barely better (perhaps lstm on word encoder and gru on sent decoder will work)
# - try putting the dropout on the rnn as well- doesn't really change the result
# - try using 300-256 as word and sent hidden
han_config = {'n_hidden_word': 128, 'n_hidden_sent': 64, 'encoder_type': 'gru', 'encoder_dropout': 0.2, 'fc_size': 0,
              'fc_dropout': 0.5, 'batchnorm': True}

register_model("cnn-emb", CNNEmb)
register_config("cnn-emb", cnn_emb_config)

register_model("rnn-emb", RNNEmb)
register_config("rnn-emb", rnn_emb_config)

register_model("han", HAN)
register_config("han", han_config)
