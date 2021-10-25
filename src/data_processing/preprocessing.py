import pandas as pd
import numpy as np

from ..utils import run


@run
def drop_empty(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Function for dropping empty rows by column value
    """
    df[column].replace('', np.nan, inplace=True)
    df.dropna(subset=[column], inplace=True)
    return df


@run
def lowercase():
    """

    """
    pass


@run
def drop_duplicate():
    pass
