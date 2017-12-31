import os
import json


def parse_model_id(model_id):
    parts = model_id.split("_")
    assert len(parts) in [2, 3], "model_id has %s pieces, should have 2 or 3" % len(parts)
    parsed = {"name": parts[0], "id": parts[1]}
    if len(parts) == 3:
        parsed["seed"] = int(parts[2])
    return parsed


def construct_model_id(model_name, id_key, seed=None):
    pieces = [model_name, str(id_key)]
    if seed is not None:
        pieces.append(str(seed))
    return "_".join(pieces)


def build_model(model_constructor, model_config):
    return model_constructor(**model_config)


class Registry(object):

    def __init__(self, config_json):
        self.model_constructors = {}
        with open(config_json, 'r') as config_json_file:
            self.model_configs = json.load(config_json_file)
        self.model_id_parts = [(name, id_key) for name in self.model_configs for id_key in self.model_configs[name]]

    def get_model_constructor(self, model_name):
        return self.model_constructors[model_name]

    def get_model_config(self, model_name, id_key):
        return self.model_configs[model_name][id_key]

    def load_model(self, model_id):
        parsed_model_id = parse_model_id(model_id)
        model_constructor = self.get_model_constructor(parsed_model_id["name"])
        model_config = self.get_model_config(parsed_model_id["name"], parsed_model_id["id"])
        return build_model(model_constructor, model_config)

    def register_model(self, model_name, model_constructor):
        self.model_constructors[model_name] = model_constructor

    def select_models(self, model_name='*', model_id_key='*', seed='*', **criteria):
        selected_models = [(name, id_key, seed) for name, id_key in self.model_id_parts]
        # Filter out our model names if we have one
        if model_name != '*':
            selected_models = [(name, id_key, seed) for name, id_key, seed in selected_models if name == model_name]
        if model_id_key != '*':
            selected_models = [(name, id_key, seed) for name, id_key, seed in selected_models if id_key == model_id_key]
        # Select all the models that match the criteria
        for criterion, value in criteria.items():
            selected_models = [(name, id_key, seed) for name, id_key, seed in selected_models if
                               self.model_configs[name][id_key][criterion] == value]

        # If the seed is * it will be used as * operator for filename
        selected_models = [construct_model_id(*parts) for parts in selected_models]
        return selected_models


registry = Registry("registry/model_configurations.json")
