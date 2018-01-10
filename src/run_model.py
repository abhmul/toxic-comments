import torch


import os
import numpy as np
import argparse

from torch.nn.functional import binary_cross_entropy_with_logits
import torch.optim as optim

from pyjet.callbacks import ModelCheckpoint, Plotter, MetricLogger
from pyjet.data import DatasetGenerator

from toxic_dataset import ToxicData
from models import load_model

import logging

parser = argparse.ArgumentParser(description='Run the models.')
parser.add_argument('-m', '--model', required=True, help='The model name to train')
parser.add_argument('-d', '--data', default='../processed_input/', help='Path to input data')
parser.add_argument('--train', action="store_true", help="Whether to run this script to train a model")
parser.add_argument('--test', action="store_true", help="Whether to run this script to generate submissions")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use when running script")
parser.add_argument('--test_batch_size', type=int, default=4, help="Batch size to use when running script")
parser.add_argument('--split', type=float, default=0.1, help="Fraction of data to split into validation")
parser.add_argument('--epochs', type=int, default=7, help="Number of epochs to train the model for")
parser.add_argument('--plot', action="store_true", help="Number of epochs to train the model for")
parser.add_argument('--seed', type=int, default=7, help="Seed fo the random number generator")
parser.add_argument('--load_model', action="store_true", help="Resumes training of the saved model.")
parser.add_argument('--use_sgd', action="store_true", help="Uses SGD instead of Adam")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SEED = args.seed
np.random.seed(SEED)

# Params
EPOCHS = args.epochs

TRAIN_ID = args.model + "_" + str(SEED)
print("Training model with id:", TRAIN_ID)
MODEL_FILE = "../models/" + TRAIN_ID + ".state"
SUBMISSION_FILE = "../submissions/" + TRAIN_ID + ".csv"
LOG_FILE = "../logs/" + TRAIN_ID + ".txt"


def train(toxic_data):
    ids, dataset = toxic_data.load_train(mode="sup")

    # Split the data
    train_data, val_data = dataset.validation_split(split=args.split, shuffle=True, seed=np.random.randint(2**32))
    # And create the generators
    traingen = DatasetGenerator(train_data, batch_size=args.batch_size, shuffle=True, seed=np.random.randint(2**32))
    valgen = DatasetGenerator(val_data, batch_size=args.batch_size, shuffle=True, seed=np.random.randint(2**32))

    # callbacks
    best_model = ModelCheckpoint(MODEL_FILE, monitor="loss", verbose=1, save_best_only=True)
    log_to_file = MetricLogger(LOG_FILE)
    callbacks = [best_model, log_to_file]
    # This will plot the losses while training
    if args.plot:
        loss_plot_fpath = '../plots/loss_' + TRAIN_ID + ".png"
        loss_plotter = Plotter(monitor='loss', scale='log', save_to_file=loss_plot_fpath, block_on_end=False)
        callbacks.append(loss_plotter)

    # Initialize the model
    model = load_model(args.model)
    if args.load_model:
        print("Loading the model to resume training")
        model.load_state(MODEL_FILE)
    # And the optimizer
    if args.use_sgd:
        optimizer = optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01, momentum=0.9)
    else:
        optimizer = optim.Adam(param for param in model.parameters() if param.requires_grad)

    # And finally train
    tr_logs, val_logs = model.fit_generator(traingen, steps_per_epoch=traingen.steps_per_epoch,
                                            epochs=EPOCHS, callbacks=callbacks, optimizer=optimizer,
                                            loss_fn=binary_cross_entropy_with_logits, validation_generator=valgen,
                                            validation_steps=valgen.steps_per_epoch)


def test(toxic_data):
    # Create the paths for the data
    ids, test_data = toxic_data.load_test()
    assert not test_data.output_labels

    # And create the generators
    testgen = DatasetGenerator(test_data, batch_size=args.test_batch_size, shuffle=False)

    # Initialize the model
    model = load_model(args.model)
    model.load_state(MODEL_FILE)

    # Get the predictions
    predictions = model.predict_generator(testgen, testgen.steps_per_epoch, verbose=1)
    ToxicData.save_submission(SUBMISSION_FILE, ids, predictions)


if __name__ == "__main__":
    # Create the paths for the data
    train_path = os.path.join(args.data, "train.npz")
    test_path = os.path.join(args.data, "test.npz")
    dictionary_path = os.path.join(args.data, "word_index.pkl")

    # Load the data
    toxic = ToxicData(train_path, test_path, dictionary_path)

    if args.train:
        train(toxic)
    if args.test:
        test(toxic)
