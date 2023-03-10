import yaml
from dotmap import DotMap
import os


def get_config_from_yml(yml_file):
    """
    Get the config from a yml file
    :param: yml_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(yml_file, "r") as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config


def process_config(yml_file):
    config = get_config_from_yml(yml_file)
    return config
