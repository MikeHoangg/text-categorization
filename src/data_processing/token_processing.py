"""
Module for processing tokens
This module works with tokens structure similar to https://spacy.io/api/token
"""
import pandas as pd

from ..utils import run


@run
def drop_not_alpha(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Function for dropping non alphabetic tokens
    """
    return df[df['is_alpha']]


@run
def drop_stop_words(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Function for dropping stop words
    """
    return df[~df['is_stop']]


@run
def drop_no_vector(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Function for dropping tokens without vectors
    """
    return df[~df['is_oov']]


@run
def drop_character(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Function for dropping single characters
    """
    return df[df['text'].str.len() > 1]


@run
def drop_fully_consonants(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Function for dropping tokens that fully consist of consonant characters
    """
    return df[~df['text'].str.fullmatch(r'^[bcdfghjklmnpqrstvwxyz]+$')]


@run
def drop_fully_vowels(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Function for dropping tokens that fully consist of vowel characters
    """
    return df[~df['text'].str.fullmatch(r'^[aeiou]+$')]
