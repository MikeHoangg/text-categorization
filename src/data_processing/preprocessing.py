"""
Module for preprocessing data
Initial filtering, data modification, etc.
"""
import pandas as pd
import numpy as np

from ..utils import run


@run
def drop_empty(df: pd.DataFrame, column: str, *args, **kwargs) -> pd.DataFrame:
    """
    Function for dropping empty rows by column value
    """
    df[column].replace('', np.nan, inplace=True)
    df.dropna(subset=[column], inplace=True)
    return df


@run
def lowercase(df: pd.DataFrame, column: str, *args, **kwargs) -> pd.DataFrame:
    """
    Function for lower casing column data
    """
    df[column] = df[column].str.lower()
    return df


@run
def drop_duplicate(df: pd.DataFrame, column: str, *args, **kwargs) -> pd.DataFrame:
    """
    Function for dropping duplicate values by column
    """
    return df.drop_duplicates(subset=[column])
