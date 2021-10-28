"""
Module for utils
"""
import os
import pandas as pd
import logging

from typing import Callable
from datetime import datetime
from pathlib import Path

# MARK: root folder
BASE_FOLDER = Path(os.getcwd()).parent.absolute()
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, 'data')


def run(func) -> Callable:
    """
    Decorator that shows running time of a function
    """

    def wrapper(*args, **kwargs):
        start = datetime.now()
        logging.info(f'{func.__name__} start: {start}')
        res = func(*args, **kwargs)
        end = datetime.now()
        logging.info(f'{func.__name__} end: {end}', f'total time: {(end - start).seconds} seconds')
        return res

    return wrapper


@run
def export(df: pd.DataFrame, file_path: str):
    """
    Function for exporting DataFrame to json file
    """
    logging.info(f'Exporting {file_path}...')
    df.to_json(file_path)
    logging.info(f'Created {file_path}')
