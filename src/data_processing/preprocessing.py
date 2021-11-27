"""
Module for preprocessing data
Initial filtering, data modification, etc.
"""
import string
import pandas as pd
import numpy as np

from typing import List

from ..utils import run
from . import BasePipe


class Preprocessor(BasePipe):
    """
    Class for preprocessing initial dataset
    """

    def __init__(self, pipeline: List[str], column: str):
        super().__init__(pipeline)
        self.column = column

    def drop_empty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for dropping empty rows by column value
        """
        df[self.column].replace('', np.nan, inplace=True)
        df.dropna(subset=[self.column], inplace=True)
        return df

    def lowercase(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for lower casing column data
        """
        df[self.column] = df[self.column].str.lower()
        return df

    def drop_duplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for dropping duplicate values by column
        """
        return df.drop_duplicates(subset=[self.column])

    def remove_punctuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for removing punctuation symbols
        """
        df[self.column] = df[self.column].str.translate(str.maketrans('', '', string.punctuation))
        return df

    @run
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Method for transforming data
        """
        return self.run_pipeline(df)[self.column]
