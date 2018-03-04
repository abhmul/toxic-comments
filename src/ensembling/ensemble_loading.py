import os
import json

from . import ensemblers


CONSTRUCTORS = {"xgboost": ensemblers.XGBEnsembler, "logreg": ensemblers.LogisticEnsembler}
CONFIG_JSON = os.path.join(os.getcwd(), "ensembling/ensembles.json")
with open(CONFIG_JSON, 'r') as config_json_file:
    CONFIGS = json.load(config_json_file)


def get_ensemble_info(ensemble_id):
    ensemble_config = CONFIGS[ensemble_id]
    constructor = CONSTRUCTORS[ensemble_config["name"]]
    submodel_names = ensemble_config["submodels"]
    params = ensemble_config["params"]
    return constructor, submodel_names, params
