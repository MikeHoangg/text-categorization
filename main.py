import os
import pandas as pd

from datetime import datetime

BASE_FOLDER = os.getcwd()
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, 'data')
PLOT_FIG_SIZE = (18, 5)


def run(func):
    def wrapper():
        start = datetime.now()
        print(f'start: {start}')
        func()
        end = datetime.now()
        print(f'end: {end}', )
        print(f'finished in: {(end - start).seconds} seconds')

    return wrapper


def export(df: pd.DataFrame, file_path: str):
    print(f'dataframe has {len(df)} rows')
    print(f'started exporting {file_path}: {datetime.now()}')
    df.to_json(file_path)
    print(f'created {file_path}: {datetime.now()}')
