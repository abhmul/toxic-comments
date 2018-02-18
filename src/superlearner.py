import argparse
import os
import logging
import json
import pandas as pd

import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

import pyjet.backend as J
from pyjet.models import SLModel
from pyjet.metrics import accuracy_with_logits
from pyjet.data import DatasetGenerator, NpDataset
from toxic_dataset import ToxicData, LABEL_NAMES
from models import load_model
from file_utils import safe_open_dir

parser = argparse.ArgumentParser(description='Train the models.')
parser.add_argument('-e', '--ensemble_id', required=True, type=str, help='The id of the ensemble to create')
parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use when running script")
parser.add_argument('--seed', type=int, default=7, help="Seed fo the random number generator")
parser.add_argument('--kfold', type=int, default=10, help="Runs kfold validation with the input number")
parser.add_argument('--create_preds', action="store_true", help="Loads the models and predicts with them")
parser.add_argument('--superlearn', action="store_true", help="Trains the superlearner")
parser.add_argument('--use_sklearn', action="store_true", help="Uses Scikit-learn's logistic regression")


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    model = SLModel(model)
    return model


def load_ensemble_configs(ensemble_json="registry/ensembles.json"):
    with open(ensemble_json, 'r') as ensemble_json_file:
        ensemble_dict = json.load(ensemble_json_file)
    return ensemble_dict


def get_ensemble_config(ensemble_dict, ensemble_id):
    return ensemble_dict[ensemble_id]


def extract_model_info(model_name):
    model_info_pieces = model_name.split("_")
    model_id = "_".join(model_info_pieces[:2])
    train_seed = int(model_info_pieces[2])
    fold_num = -1
    if len(model_info_pieces) > 3:
        # Remove the word fold from the string
        fold_num = int(model_info_pieces[3][4:])
    return model_id, train_seed, fold_num


def expand_model_names(model_names, k):
    expanded_names = [[] for _ in model_names]
    for i, name in enumerate(model_names):
        for fold in range(k):
            expanded_names[i].append(name + "_fold%s" % fold)
    return expanded_names


def load_dataset(data_path):
    train_path = os.path.join(data_path, "train.npz")
    test_path = os.path.join(data_path, "test.npz")
    dictionary_path = os.path.join(data_path, "word_index.pkl")
    # Load the data
    toxic = ToxicData(train_path, test_path, dictionary_path)
    train_ids, train_dataset = toxic.load_train(mode="sup")
    return train_dataset


def predict_val(model_foldset, data_path, batch_size, k, seed=7, pred_Y=None):
    dataset = load_dataset(data_path)
    logging.info("Loaded dataset from %s" % data_path)

    # Initialize the predictions array
    pred_X = np.zeros((len(dataset), len(LABEL_NAMES)), dtype=np.float32)
    # Used to know whether to check if the y's match up, or to build the y's
    if pred_Y is None:
        check_ys = False
        pred_Y = np.zeros((len(dataset), len(LABEL_NAMES)), dtype=np.float32)
    else:
        check_ys = True

    # Extract the model
    model_id = extract_model_info(model_foldset[0])[0]
    logging.info("Loading model %s" % model_id)
    model = load_model(model_id)

    cur_sample_ind = 0
    # Set the random seed so we get the same folds
    np.random.seed(seed)
    for i, (train_data, val_data) in enumerate(dataset.kfold(k=k, shuffle=True, seed=np.random.randint(2 ** 32))):
        logging.info("Getting train predictions for fold%s" % i)
        model_file = os.path.join("../models/", model_foldset[i] + ".state")
        model.load_state(model_file)
        logging.info("Loaded weights from %s" % model_file)
        model_id_temp, train_seed, fold_num = extract_model_info(model_foldset[i])

        # Some sanity checks
        assert train_seed == seed
        assert fold_num == i
        assert model_id_temp == model_id

        val_data.output_labels = False
        valgen = DatasetGenerator(val_data, batch_size=batch_size, shuffle=False)

        # Predict on the validation set
        if not check_ys:
            logging.info("Constructing ys")
            # For the first model, we'll gather the labels
            pred_Y[cur_sample_ind:cur_sample_ind + len(val_data)] = val_data.y
        else:
            logging.info("Checking that ys are correct")
            # For the other models we'll check to make sure the labels are the same
            assert np.all(pred_Y[cur_sample_ind:cur_sample_ind + len(val_data)] == val_data.y)
        pred_X[cur_sample_ind:cur_sample_ind + len(val_data)] = model.predict_generator(valgen,
                                                                                        prediction_steps=valgen.steps_per_epoch,
                                                                                        verbose=1)
        cur_sample_ind += len(val_data)

    assert cur_sample_ind == len(dataset)
    # Clean up cuda memory
    del model
    if J.use_cuda:
        torch.cuda.empty_cache()

    return pred_X, pred_Y


def create_preds(model_names, data_paths, batch_size, k, seed=7, savedir="../superlearner/"):
    num_base_learners = len(model_names)
    assert num_base_learners == len(data_paths)
    logging.info("Using %s base learners" % num_base_learners)
    expanded_model_names = expand_model_names(model_names, k)
    # Build the new train data to train the meta learner on
    pred_X, pred_Y = None, None
    for j, model_foldset in enumerate(expanded_model_names):
        # Try to load it, otherwise create the predictions
        try:
            single_pred_X, pred_Y = load_predictions(model_names[j], savedir, pred_Y=pred_Y)
        except IOError:
            # If the file is not there, create it
            logging.info("Couldn't find predictions for " + model_names[j] + ", creating instead")
            single_pred_X, pred_Y = predict_val(model_foldset, data_paths[j], batch_size, k, seed=seed, pred_Y=pred_Y)
            save_predictions(single_pred_X, pred_Y, model_names[j], savedir)

        assert single_pred_X.ndim == 2
        # Construct the X array if this is our first iteration
        if j == 0:
            pred_X = np.zeros((single_pred_X.shape[0], num_base_learners, single_pred_X.shape[1]), dtype=np.float32)

        assert pred_X.shape[0] == single_pred_X.shape[0]
        assert pred_X.shape[2] == single_pred_X.shape[1]
        pred_X[:, j] = single_pred_X

    return pred_X, pred_Y


def save_predictions(pred_X, pred_Y, model_name, pred_savedir):
    savepath = os.path.join(pred_savedir, model_name + ".npz")
    np.savez(savepath, X=pred_X, Y=pred_Y)


def load_predictions(model_name, pred_savedir, pred_Y=None):
    savepath = os.path.join(pred_savedir, model_name + ".npz")
    logging.info("Loading predictions from " + savepath)
    preds = np.load(savepath)
    # Load the predictions
    pred_X = preds["X"]
    if pred_Y is None:
        pred_Y = preds["Y"]
    else:
        assert np.all(pred_Y == preds["Y"])
    return pred_X, pred_Y


def train_superlearner(pred_X, pred_Y):
    # Now train 6 dense layers
    num_base_learners = pred_X.shape[1]
    weights = np.zeros((num_base_learners, len(LABEL_NAMES)))
    for i, label in enumerate(LABEL_NAMES):
        logging.info("Training logistic regression for label %s" % label)
        pred_dataset = NpDataset(x=pred_X[:, :, i], y=pred_Y[:, i:i + 1])
        datagen = DatasetGenerator(pred_dataset, batch_size=len(pred_dataset), shuffle=False)
        logistic_reg = build_model(num_base_learners, 1)
        optimizer = torch.optim.SGD(logistic_reg.parameters(), lr=0.01)
        train_logs, val_logs = logistic_reg.fit_generator(datagen, steps_per_epoch=datagen.steps_per_epoch, epochs=1000,
                                                          optimizer=optimizer,
                                                          loss_fn=F.binary_cross_entropy_with_logits,
                                                          metrics=[accuracy_with_logits], verbose=0)
        logging.info("Final Loss: %s" % train_logs["loss"][-1])
        logging.info("Final Accuracy: %s" % train_logs["accuracy_with_logits"][-1])
        weight = logistic_reg.torch_module.linear.weight.data
        weights[:, i] = weight.cpu().numpy().flatten()
        logging.info("Trained weights: {}".format(weights[:, i]))
    return weights


def train_superlearner_sklearn(pred_X, pred_Y):
    # Now train 6 dense layers
    num_base_learners = pred_X.shape[1]
    weights = np.zeros((num_base_learners, len(LABEL_NAMES)))
    for i, label in enumerate(LABEL_NAMES):
        logging.info("Training logistic regression for label %s" % label)
        X = pred_X[:, :, i]
        Y = pred_Y[:, i]
        logistic_reg = LogisticRegression(C=1e32, fit_intercept=False, max_iter=100000)
        logistic_reg.fit(X, Y)
        # For scoring
        loss = log_loss(Y, logistic_reg.predict_proba(X))
        acc = accuracy_score(Y, logistic_reg.predict(X))
        logging.info("Final Loss: %s" % loss)
        logging.info("Final Accuracy: %s" % acc)
        weight = logistic_reg.coef_
        weights[:, i] = weight.flatten()
        logging.info("Trained weights: {}".format(weights[:, i]))
    return weights


def ensemble_submissions(submission_fnames, weights):
    assert len(submission_fnames) > 0, "Must provide at least one submission to ensemble."
    # Check that we have a weight for each submission
    assert len(submission_fnames) == len(weights), "Number of submissions and weights must match."
    # Get the id column of the submissions
    ids = pd.read_csv(submission_fnames[0])['id'].values
    # Read in all the submission values
    submissions = [pd.read_csv(sub_fname)[LABEL_NAMES].values for sub_fname in submission_fnames]
    # Combine them based on their respective weights
    combined = 0
    for j, sub in enumerate(submissions):
        if np.all((0 <= sub) & (sub <= 1.)):
            logging.info("Applying logit to submission %s" % submission_fnames[j])
            sub = logit(sub)
        combined = combined + weights[j][np.newaxis] * sub
    # combined = expit(combined)
    return ids, combined


if __name__ == "__main__":
    args = parser.parse_args()
    SEED = args.seed
    np.random.seed(SEED)
    logging.info("Opening the ensemble configs")
    ensemble_config_dict = load_ensemble_configs()
    ensemble_config = get_ensemble_config(ensemble_config_dict, args.ensemble_id)
    pred_savepath = safe_open_dir("../superlearner_preds/")
    # Get the predictions
    pred_x, pred_y = create_preds(ensemble_config["files"], ensemble_config["data"], batch_size=args.batch_size,
                                  k=args.kfold, seed=SEED, savedir=pred_savepath)

    # Train the meta-learner
    if args.superlearn:
        if args.use_sklearn:
            logging.info("Training superlearner with scikit-learn")
            weights = train_superlearner_sklearn(pred_x, pred_y)
        else:
            weights = train_superlearner(pred_x, pred_y)

        # Run the ensembling
        submission_fnames = [os.path.join("../submissions/", fname + ".csv") for fname in ensemble_config["files"]]
        logging.info("Using submission_fnames: " + str(submission_fnames))
        test_ids, combined_preds = ensemble_submissions(submission_fnames, weights)
        logging.info("Combined preds shape: {}".format(combined_preds.shape))
        ToxicData.save_submission(os.path.join("../submissions/", "superlearner_" + args.ensemble_id + ".csv"),
                                  test_ids, combined_preds)
