import os
import logging
import sys
sys.path.append('..')

import torch
import numpy as np

import pyjet.backend as J
from pyjet.data import DatasetGenerator, NpDataset
from toxic_dataset import ToxicData, LABEL_NAMES

from models import load_model
from registry import model_registry
from file_utils import safe_open_dir


def load_dataset(data_path):
    train_path = os.path.join(data_path, "train.npz")
    test_path = os.path.join(data_path, "test.npz")
    dictionary_path = os.path.join(data_path, "word_index.pkl")
    # Load the data
    toxic = ToxicData(train_path, test_path, dictionary_path)
    train_ids, train_data = toxic.load_train(mode="sup")
    return train_data


def fix_pyjet_model_id(model_id, fold_num):
    return os.path.join("../models/", model_id + "_fold{fold_num}.state".format(fold_num=fold_num))


def predict_val(model_name, train_data, k, seed=7, Y=None, batch_size=32):
    """
    Creates a k-fold validation set for the next level of ensembling. These
    are the predictions from each submodel in kfold model on its held out fold.
    :param model_name: The name of the model to use to predict
    :param train_data: The numpy array holding the train_data
    :param k: The number of folds the model was trained with
    :param ids: The ids for the prediction data (Optional, if passed will verify the ids match up)
    :param Y: The labels for the prediction data (Optional, if passed will verify the labels match up)
    :param pyjet: Whether or not the input model is a pyjet model or not
    :param batch_size: If using a pyjet model, the batch size to use
    :return: The ids, predictions, and labels for the train data
    """
    pyjet = model_name in model_registry.registry
    if pyjet:
        predictions, pred_labels = predict_val_pyjet(model_name, train_data, k, seed=seed, batch_size=batch_size)
    else:
        raise NotImplementedError("Using other types of models is not implemented yet")

    # Used to know whether to check if the y's match up
    if Y is not None:
        assert np.all(Y == pred_labels)

    return predictions, pred_labels


def predict_val_pyjet(model_name, dataset, k, seed=7, batch_size=32):

    # Extract the model
    parsed = model_registry.parse_model_id(model_name)
    model_class, id_key = parsed["name"], parsed["id"]
    model_id = model_registry.construct_model_id(model_class, id_key, seed=seed)
    logging.info("Loading PyJet model %s" % model_id)
    model = load_model(model_id)

    # Initialize the arrays
    predictions = np.zeros((len(dataset), len(LABEL_NAMES)), dtype=np.float32)
    pred_ids = np.zeros_like(dataset)
    pred_labels = np.zeros((len(dataset), len(LABEL_NAMES)), dtype=np.float32)

    cur_sample_ind = 0
    # Set the random seed so we get the same folds
    np.random.seed(seed)
    for i, (train_data, val_data) in enumerate(dataset.kfold(k=k, shuffle=True, seed=np.random.randint(2 ** 32))):
        logging.info("Getting train predictions for fold%s" % i)
        model_file = fix_pyjet_model_id(model_id, i)
        model.load_state(model_file)
        logging.info("Loaded weights from %s" % model_file)

        val_data.output_labels = False
        valgen = DatasetGenerator(val_data, batch_size=batch_size, shuffle=False)

        # Get the labels on the validation set
        pred_labels[cur_sample_ind:cur_sample_ind + len(val_data)] = val_data.y
        # Get the ids of the validation set
        pred_ids[cur_sample_ind:cur_sample_ind + len(val_data)] = val_data.ids
        # Get our predictions on the validation set
        predictions[cur_sample_ind:cur_sample_ind + len(val_data)] = model.predict_generator(valgen,
                                                                                             prediction_steps=valgen.steps_per_epoch,
                                                                                             verbose=1)
        cur_sample_ind += len(val_data)

    assert cur_sample_ind == len(dataset)
    # Clean up cuda memory
    del model
    if J.use_cuda:
        torch.cuda.empty_cache()

    return predictions, pred_labels


def load_predictions(model_name, pred_savedir, pred_labels=None):
    savepath = os.path.join(pred_savedir, model_name + ".npz")
    logging.info("Loading predictions from " + savepath)
    preds = np.load(savepath)
    # Load the predictions
    predictions = preds["X"]

    # Some checks to make sure our data is not corrupt
    if pred_labels is None:
        pred_labels = preds["Y"]
    else:
        assert np.all(pred_labels == preds["Y"])

    return predictions, pred_labels


def save_predictions(predictions, pred_labels, model_name, pred_savedir):
    savepath = os.path.join(safe_open_dir(pred_savedir), model_name + ".npz")
    np.savez(savepath, X=predictions, Y=pred_labels)
    logging.info("Saved validation preds to {}".format(savepath))


def create_predictions(model_names, k, seed=7, savedir="../superlearner_preds/", data_paths=tuple(), batch_size=32):
    num_base_learners = len(model_names)
    logging.info("Using %s base learners" % num_base_learners)
    # Build the new train data to train the meta learner on
    predictions, pred_labels = None, None
    for j, model_name in enumerate(model_names):
        # Try to load it, otherwise create the predictions
        try:
            single_predictions, pred_labels = load_predictions(model_name, savedir, pred_labels=pred_labels)
        except:
            # If the file is not there, create it
            logging.info("Couldn't load predictions for " + model_name + ", creating instead")
            train_data = load_dataset(data_paths[j])
            single_predictions, pred_labels = predict_val(model_name, train_data, k, seed=seed,
                                                                     Y=pred_labels, batch_size=batch_size)
            save_predictions(single_predictions, predictions, model_names[j], savedir)

        assert single_predictions.ndim == 2
        # Construct the X array if this is our first iteration
        if j == 0:
            predictions = np.zeros((single_predictions.shape[0], num_base_learners, single_predictions.shape[1]),
                                   dtype=np.float32)

        assert predictions.shape[0] == single_predictions.shape[0]
        assert predictions.shape[2] == single_predictions.shape[1]
        predictions[:, j] = single_predictions

    return NpDataset(predictions, y=pred_labels)
