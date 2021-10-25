import os
import pandas as pd

from typing import Callable
from datetime import datetime
from pathlib import Path

BASE_FOLDER = Path(os.getcwd()).parent.absolute()  # root folder
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, 'data')


def run(func) -> Callable:
    """
    Decorator that shows running time of a function
    """

    def wrapper(*args, **kwargs):
        start = datetime.now()
        print(f'{func.__name__} start: {start}')
        res = func(*args, **kwargs)
        end = datetime.now()
        print(f'{func.__name__} end: {end}', f'total time: {(end - start).seconds} seconds')
        return res

    return wrapper


@run
def export(df: pd.DataFrame, file_path: str):
    """
    Function for exporting DataFrame to json file
    """
    print(f'DataFrame has {len(df.index)} rows')
    print(f'Exporting {file_path}...')
    df.to_json(file_path)
    print(f'Created {file_path}')
