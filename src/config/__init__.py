import yaml
from trafaret import DataError

from validation import CONFIG_FILE_VALIDATOR
from ..errors import BrokenConfigException


def load_config(config_path: str):
    with open(config_path) as config_file:
        config_data = yaml.safe_load(config_file)
    try:
        CONFIG_FILE_VALIDATOR.check(config_data)
    except DataError as ex:
        raise BrokenConfigException(ex.value)
