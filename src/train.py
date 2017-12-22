import os
import numpy as np
import pandas as pd
import argparse

from pyjet.callbacks import ModelCheckpoint, Plotter
from pyjet.data import DatasetGenerator
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.optim as optim

from toxic_dataset import ToxicData, LABEL_NAMES
from models import load_model, load_config

SEED = 113
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Train the models.')
parser.add_argument('-m', '--model', default="cnn-emb", help='The model name to train')
parser.add_argument('-d', '--data', default="../processed_input",
                    help='Path to input data')
parser.add_argument('--train', action="store_true", help="Whether to run this script to train a model")
parser.add_argument('--test', action="store_true", help="Whether to run this script to generate submissions")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use when running script")
parser.add_argument('--test_batch_size', type=int, default=8, help="Batch size to use when running script")
parser.add_argument('--split', type=float, default=0.1, help="Fraction of data to split into validation")
parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train the model for")
parser.add_argument('--plot', action="store_true", help="Number of epochs to train the model for")
args = parser.parse_args()

# Params
MODEL_FUNC = load_model(args.model)
CONFIG = load_config(args.model)

EPOCHS = args.epochs
model_file_id = args.model + "_" + "_".join(k + "_" + str(v) for k, v in sorted(CONFIG.items()))
MODEL_FILE = "../models/" + model_file_id + ".state"
SUBMISSION_FILE = "../submissions/" + model_file_id + ".csv"
LOG_FILE = "../logs/" + model_file_id + ".txt"
PARSED_SENT = args.model[:3] == "han"


def train(toxic_data):
    ids, dataset = toxic_data.load_train(mode="sup")
    dataset = toxic_data.embed(dataset, PARSED_SENT)

    # Split the data
    train_data, val_data = dataset.validation_split(split=args.split, shuffle=True, seed=np.random.randint(2**32))
    # And create the generators
    traingen = DatasetGenerator(train_data, batch_size=args.batch_size, shuffle=True, seed=np.random.randint(2**32))
    valgen = DatasetGenerator(val_data, batch_size=args.batch_size, shuffle=True, seed=np.random.randint(2**32))

    # callbacks
    best_model = ModelCheckpoint(
        MODEL_FILE, monitor="loss", verbose=1, save_best_only=True)
    callbacks = [best_model]
    # This will plot the losses while training
    if args.plot:
        loss_plot_fpath = '../plots/loss_' + model_file_id + ".png"
        loss_plotter = Plotter(monitor='loss', scale='log', save_to_file=loss_plot_fpath)
        callbacks.append(loss_plotter)

    # Initialize the model
    model = MODEL_FUNC(**CONFIG)
    # And the optimizer
    optimizer = optim.Adam(model.parameters())

    # And finally train
    tr_logs, val_logs = model.fit_generator(traingen, steps_per_epoch=traingen.steps_per_epoch,
                                            epochs=EPOCHS, callbacks=callbacks, optimizer=optimizer,
                                            loss_fn=binary_cross_entropy_with_logits, validation_generator=valgen,
                                            validation_steps=valgen.steps_per_epoch)
    # Save the output
    if "loss" in val_logs.logs:
        val_losses = val_logs["loss"]
        with open(LOG_FILE, 'w') as logs:
            for loss in val_losses:
                print(loss, file=logs)


def save_submission(pred_ids, predictions):
    submid = pd.DataFrame({'id': pred_ids})
    submission = pd.concat([submid, pd.DataFrame(data=predictions, columns=LABEL_NAMES)], axis=1)
    submission.to_csv(SUBMISSION_FILE, index=False)


def test(toxic_data):
    # Create the paths for the data
    ids, test_data = toxic_data.load_test()
    assert not test_data.output_labels
    test_data = toxic_data.embed(test_data, PARSED_SENT)

    # And create the generators
    testgen = DatasetGenerator(test_data, batch_size=args.test_batch_size, shuffle=False)

    # Initialize the model
    model = MODEL_FUNC(**CONFIG)
    model.load_state(MODEL_FILE)

    # Get the predictions
    predictions = model.predict_generator(testgen, testgen.steps_per_epoch, verbose=1)
    save_submission(ids, predictions)


if __name__ == "__main__":
    # Create the paths for the data
    train_path = os.path.join(args.data, "train.npz")
    test_path = os.path.join(args.data, "test.npz")
    embeddings_path = os.path.join(args.data, "embeddings.npy")
    dictionary_path = os.path.join(args.data, "word_to_id.pkl")

    # Load the data
    toxic = ToxicData(train_path, test_path, embeddings_path, dictionary_path)

    if args.train:
        train(toxic)
    if args.test:
        test(toxic)
