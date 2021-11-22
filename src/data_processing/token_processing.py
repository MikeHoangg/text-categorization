"""
Module for processing tokens
"""
import re
import os
import itertools
import pandas as pd
import numpy as np
import spacy

from multiprocessing import Pool
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from spacy.tokens import Token

from ..utils import run


class SpacyTokenNormalizer(BaseEstimator, TransformerMixin):
    """
    This class works with tokens structure similar to https://spacy.io/api/token
    """
    def __init__(self, pipeline: List[str], spacy_core: str = 'en_core_web_lg'):
        self.spacy_core = spacy_core
        self.pipeline = pipeline
        self._spacy_processor = spacy.load(spacy_core)

    @run
    def drop_not_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping non alphabetic tokens
        """
        return df[df['is_alpha']]

    @run
    def drop_stop_words(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping stop words
        """
        return df[~df['is_stop']]

    @run
    def drop_no_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping tokens without vectors
        """
        return df[df['has_vector']]

    @run
    def drop_character(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping single characters
        """
        return df[df['text'].str.len() > 1]

    @run
    def drop_fully_consonants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping tokens that fully consist of consonant characters
        """
        return df[~df['text'].str.fullmatch(r'^[bcdfghjklmnpqrstvwxyz]+$', flags=re.IGNORECASE)]

    @run
    def drop_fully_vowels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping tokens that fully consist of vowel characters
        """
        return df[~df['text'].str.fullmatch(r'^[aeiou]+$', flags=re.IGNORECASE)]

    def _tokenize(self, sentence: str) -> List[dict]:
        """
        Method for tokenizing sentence
        """

        def __token_modifier(token: Token) -> dict:
            """
            Method for modifying spacy tokens into basic type
            """
            return {
                'text': token.text,
                'is_stop': token.is_stop,
                'is_alpha': token.is_alpha,
                'has_vector': not token.is_oov,
                'vector': token.vector,
                'vector_norm': token.vector_norm,
            }

        tokens = self._spacy_processor(sentence)
        return [__token_modifier(token) for token in tokens]

    def _get_tokens(self, data: List[str]) -> pd.DataFrame:
        """
        Method for getting separate tokens and filtering them
        """
        tokens = list(itertools.chain.from_iterable([self._tokenize(sentence) for sentence in data]))
        return pd.DataFrame.from_records(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, data: List[str]) -> pd.DataFrame:
        """
        Method for transforming data
        """
        with Pool() as pool:
            token_dataframes = pool.map(self._get_tokens, np.array_split(data, os.cpu_count()))
            df = pd.concat(token_dataframes)
        for pipe in self.pipeline:
            df = getattr(self, pipe)(df)
        return df
