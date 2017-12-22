import os
import numpy as np
import argparse

from pyjet.callbacks import ModelCheckpoint, Plotter
from pyjet.data import DatasetGenerator
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.optim as optim
from pyjet.metrics import accuracy

from toxic_dataset import ToxicData
import models

SEED = 113
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Train the models.')
parser.add_argument('-m', '--model', default="cnn-emb", help='The model name to train')
parser.add_argument('-d', '--data', default="../processed_input",
                    help='Path to input data')
args = parser.parse_args()

# Params
cnn_emb_params = {'num_features': 300, 'kernel_size': 3, 'pool_kernel_size': 2, 'pool_stride': 1, 'n1_filters': 128,
                  'n2_filters': 64, 'k': 5, 'conv_dropout': 0.2, 'fc_dropout': 0.5, 'batchnorm': True}
PARAMS = {'cnn-emb': cnn_emb_params}
MODELS = {'cnn-emb': models.CNNEmb}

MODEL_FUNC = MODELS[args.model]
PARAM_SETTINGS = PARAMS[args.model]

EPOCHS = 20
model_file_id = args.model + "_" + "_".join(k + "_" + str(v) for k, v in sorted(PARAM_SETTINGS.items()))
MODEL_FILE = "../models/" + model_file_id + ".state"

# Create the paths for the data
train_path = os.path.join(args.data, "train.npz")
test_path = os.path.join(args.data, "test.npz")
embeddings_path = os.path.join(args.data, "embeddings.npy")
dictionary_path = os.path.join(args.data, "word_to_id.pkl")

# Load the data
toxic = ToxicData(train_path, test_path, embeddings_path, dictionary_path)
word_to_id, id_to_word = toxic.load_dictionary()
# logger = AttentionLogger(id_to_word, log_every=1000, pad=False)
logger = None
ids, dataset = toxic.load_train(mode="sup")
dataset = toxic.embed(dataset)


# Split the data
train_data, val_data = dataset.validation_split(split=0.1, shuffle=True, seed=np.random.randint(1000))
# And create the generators
traingen = DatasetGenerator(train_data, batch_size=32, shuffle=True, seed=np.random.randint(1000))
valgen = DatasetGenerator(val_data, batch_size=32, shuffle=True, seed=np.random.randint(1000))

# callbacks
best_model = ModelCheckpoint(
    MODEL_FILE, monitor="loss", verbose=1, save_best_only=True)
# This will plot the losses while training
loss_plot_fpath = '../plots/loss_' + model_file_id + ".png"
loss_plotter = Plotter(monitor='loss', scale='log', save_to_file=loss_plot_fpath)
callbacks = [best_model, loss_plotter]

# Initialize the model
model = MODEL_FUNC(**PARAM_SETTINGS)
# And the optimizer
optimizer = optim.Adam(model.parameters())

# And finally train
model.fit_generator(traingen, steps_per_epoch=traingen.steps_per_epoch,
                    epochs=EPOCHS, callbacks=callbacks, optimizer=optimizer,
                    loss_fn=binary_cross_entropy_with_logits, validation_generator=valgen,
                    validation_steps=valgen.steps_per_epoch)
