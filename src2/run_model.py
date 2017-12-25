import os
from pathlib import Path
import numpy as np
import argparse

from pyjet.callbacks import ModelCheckpoint, Plotter, MetricLogger
from pyjet.data import DatasetGenerator
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.optim as optim

from toxic_dataset import ToxicData
from models import load_model, load_config

parser = argparse.ArgumentParser(description='Run the models.')
parser.add_argument('-m', '--model', required=True, help='The model name to train')
parser.add_argument('-d', '--data', '../processed_input/', help='Path to input data')
parser.add_argument('-e', '--embeddings', default='../embeddings/', help='Path to directory with embeddings to use')
parser.add_argument('--train', action="store_true", help="Whether to run this script to train a model")
parser.add_argument('--test', action="store_true", help="Whether to run this script to generate submissions")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use when running script")
parser.add_argument('--test_batch_size', type=int, default=8, help="Batch size to use when running script")
parser.add_argument('--split', type=float, default=0.1, help="Fraction of data to split into validation")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model for")
parser.add_argument('--plot', action="store_true", help="Number of epochs to train the model for")
parser.add_argument('--seed', type=int, default=113, help="Seed fo the random number generator")
args = parser.parse_args()

SEED = args.seed
np.random.seed(SEED)

# Params
MODEL_FUNC = load_model(args.model)
CONFIG = load_config(args.model)
EPOCHS = args.epochs


def get_train_id(model_name, embeddings_path, config, seed):
    embedding_name = Path(embeddings_path).parts[-1]
    items_of_id = [model_name, embedding_name] + [k + "-" + str(v) for k, v in sorted(config.items())] + [str(seed)]
    return "_".join(items_of_id)


TRAIN_ID = get_train_id(args.model, args.embeddings, CONFIG, SEED)
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
    model = MODEL_FUNC(args.embeddings, **CONFIG)
    # And the optimizer
    optimizer = optim.Adam(model.parameters())

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
    model = MODEL_FUNC(args.embeddings, **CONFIG)
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
