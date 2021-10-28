"""
Module with core functions for processing data
"""
import pandas as pd

from ..utils import run


@run
def count_words(tokens: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Function that creates DataFrame of token count
    tokens DataFrame example:
                  word
        0          abc
        1          abc
        2      example
              ...
    result DataFrame example:
                 word  count
        0         abc  2
        1     example  1
              ...   ...
    """
    count_series = pd.Series([token for token in tokens['text']]).value_counts()
    return pd.DataFrame({'word': count_series.index, 'count': count_series.values})


@run
def cluster_words(*args, **kwargs):
    pass


@run
def create_core(*args, **kwargs):
    pass
