import glob
from os.path import dirname, basename, isfile

from registry import registry

from models import cnn_emb, rnn_emb, han, dense_rnn_emb, mha_rnn_emb

modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]


def load_model(model_id):
    return registry.load_model(model_id)
