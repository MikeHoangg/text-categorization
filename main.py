import os

import pandas as pd

from typing import Callable
from datetime import datetime

BASE_FOLDER = os.getcwd()
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, 'data')


def run(func) -> Callable:
    """
    Decorator that shows running time of a process
    :param func: wrapped function
    :return:
    """
    def wrapper():
        start = datetime.now()
        print(f'start: {start}')
        func()
        end = datetime.now()
        print(f'end: {end}', )
        print(f'finished in: {(end - start).seconds} seconds')

    return wrapper


def export(df: pd.DataFrame, file_path: str):
    """
    Function for exporting DataFrame to json file
    :param df: DataFrame for exporting
    :param file_path: path of export file
    :return:
    """
    print(f'dataframe has {len(df)} rows')
    print(f'started exporting {file_path}: {datetime.now()}')
    df.to_json(file_path)
    print(f'created {file_path}: {datetime.now()}')
