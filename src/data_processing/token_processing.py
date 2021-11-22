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

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import tokenize
from spacy.tokens import Token

from ..utils import run
from . import BasePipe


class SpacyTokenNormalizer(BasePipe):
    """
    This class works with tokens structure similar to https://spacy.io/api/token
    """

    def __init__(self, pipeline: List[str], spacy_core: str = 'en_core_web_lg'):
        super().__init__(pipeline)
        self.spacy_core = spacy_core
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

    @run
    def transform(self, data: List[str]) -> pd.DataFrame:
        """
        Method for transforming data
        """
        with Pool() as pool:
            token_dataframes = pool.map(self._get_tokens, np.array_split(data, os.cpu_count()))
            df = pd.concat(token_dataframes)
        return self.run_pipeline(df)


class TokenNormalizer(BasePipe):
    """
    Class for tokenization using gensim
    """

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

        def __token_modifier(token: str) -> dict:
            """
            Method for modifying spacy tokens into basic type
            """
            return {
                'text': token,
                'is_stop': token in STOPWORDS,
                # todo: add implementation
                'is_alpha': ...,
            }

        tokens = list(tokenize(sentence))
        tokens = [__token_modifier(token) for token in tokens]
        tokens = [
            TaggedDocument([token['text']], ['d{}'.format(idx)])
            for idx, token in enumerate(tokens)
        ]
        model = Doc2Vec(tokens, min_count=0)
        for token, vect in zip(tokens, model.docvecs):
            token['vector'] = vect
            # todo: add implementation
            token['has_vector'] = ...
            token['vector_norm'] = ...
        return tokens

    def _get_tokens(self, data: List[str]) -> pd.DataFrame:
        """
        Method for getting separate tokens and filtering them
        """
        tokens = list(itertools.chain.from_iterable([self._tokenize(sentence) for sentence in data]))
        return pd.DataFrame.from_records(tokens)

    def transform(self, data: List[str]) -> pd.DataFrame:
        """
        Method for transforming data
        """
        with Pool() as pool:
            token_dataframes = pool.map(self._get_tokens, np.array_split(data, os.cpu_count()))
            df = pd.concat(token_dataframes)
        return self.run_pipeline(df)
