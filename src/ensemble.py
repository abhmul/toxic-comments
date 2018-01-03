import argparse
import os
import logging
import json
import pandas as pd

from toxic_dataset import ToxicData, LABEL_NAMES

parser = argparse.ArgumentParser(description='Train the models.')
parser.add_argument('-e', '--ensemble_id', required=True, type=str, help='The id of the ensemble to create')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_ensemble_configs(ensemble_json="registry/ensembles.json"):
    with open(ensemble_json, 'r') as ensemble_json_file:
        ensemble_dict = json.load(ensemble_json_file)
    return ensemble_dict


def get_ensemble_config(ensemble_dict, ensemble_id):
    return ensemble_dict[ensemble_id]


def ensemble_submissions(submission_fnames, weights=None):
    assert len(submission_fnames) > 0, "Must provide at least one submission to ensemble."

    if weights is None:
        weights = [1 / len(submission_fnames)] * len(submission_fnames)
    # Check that we have a weight for each submission
    assert len(submission_fnames) == len(weights), "Number of submissions and weights must match."
    # Normalize the weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    # Get the id column of the submissions
    ids = pd.read_csv(submission_fnames[0])['id'].values
    # Read in all the submission values
    submissions = [pd.read_csv(sub_fname)[LABEL_NAMES].values for sub_fname in submission_fnames]
    # Combine them based on their respective weights
    combined = 0
    for weight, sub in zip(weights, submissions):
        combined = combined + weight * sub
    return ids, combined


if __name__ == "__main__":
    # Load the ensemble config
    logging.info("Opening the ensemble configs")
    ensemble_config_dict = load_ensemble_configs()
    ensemble_config = get_ensemble_config(ensemble_config_dict, args.ensemble_id)
    # Gather the filenames
    submission_fnames = [os.path.join("../submissions/", fname + ".csv") for fname in ensemble_config["files"]]
    logging.info("Files: {}".format(submission_fnames))
    # Ensemble the submissions
    ids, combined = ensemble_submissions(submission_fnames, weights=ensemble_config["weights"])
    ToxicData.save_submission(os.path.join("../submissions/", "ensemble_" + args.ensemble_id + ".csv"), ids, combined)
