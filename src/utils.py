"""
Module for utils
"""
import os
import pandas as pd
import logging

from typing import Callable
from pathlib import Path

# MARK: root folder
BASE_FOLDER = Path(os.getcwd()).parent.absolute()
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, 'data')


def run(func) -> Callable:
    """
    Decorator that shows running time of a function
    """

    def wrapper(*args, **kwargs):
        logging.info(f'{func.__repr__()} start.')
        res = func(*args, **kwargs)
        logging.info(f'{func.__repr__()} end.')
        return res

    return wrapper


@run
def export_json(df: pd.DataFrame, file_path: str):
    """
    Function for exporting DataFrame to json file
    """
    logging.info(f'Exporting {file_path}...')
    df.to_json(file_path)
    logging.info(f'Created {file_path}')


@run
def export_csv(df: pd.DataFrame, file_path: str):
    """
    Function for exporting DataFrame to csv file
    """
    logging.info(f'Exporting {file_path}...')
    df.to_csv(file_path)
    logging.info(f'Created {file_path}')
