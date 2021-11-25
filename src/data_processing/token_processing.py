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

from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import tokenize
from gensim import models

from ..utils import run
from . import BasePipe


class TokenFilteringMixin:
    STOPWORDS = ...

    def _vectorize(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def vectorize(self, df: pd.DataFrame) -> pd.DataFrame:
        with Pool() as pool:
            tokens_dataframes = pool.map(self._vectorize, np.array_split(df, os.cpu_count()))
            df = pd.concat(tokens_dataframes)
        return df

    def drop_not_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping non alphabetic tokens
        """
        return df[df['text'].str.isalpha()]

    def drop_stop_words(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping stop words
        """
        return df[~df['text'].isin(self.STOPWORDS)]

    def drop_no_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping tokens without vectors
        """
        return df[df['has_vector']]

    def drop_character(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping single characters
        """
        return df[df['text'].str.len() > 1]

    def drop_fully_consonants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping tokens that fully consist of consonant characters
        """
        return df[~df['text'].str.fullmatch(r'^[bcdfghjklmnpqrstvwxyz]+$', flags=re.IGNORECASE)]

    def drop_fully_vowels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for dropping tokens that fully consist of vowel characters
        """
        return df[~df['text'].str.fullmatch(r'^[aeiou]+$', flags=re.IGNORECASE)]

    def count_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that creates DataFrame of token count
        tokens DataFrame example:
                text    ...
            0   abc     ...
            1   abc     ...
            2   example ...
                ...     ...
        result DataFrame example:
                text    ... count
            0   abc     ... 2
            1   example ... 1
                ...     ... ...
        """
        count_series = df['text'].value_counts()
        df = df.drop_duplicates(subset=['text'])
        df['count'] = df.apply(lambda token: count_series.at[token['text']], axis=1)
        return df

    @run
    def transform(self, data: List[str]) -> pd.DataFrame:
        """
        Method for transforming data
        """
        tokens = list(itertools.chain.from_iterable([tokenize(sentence) for sentence in data]))
        df = pd.DataFrame(tokens, columns=['text'])
        return self.run_pipeline(df)


class SpacyTokenProcessor(TokenFilteringMixin, BasePipe):
    """
    This class works with tokens structure similar to https://spacy.io/api/token
    """

    def __init__(self, pipeline: List[str], spacy_core: str = 'en_core_web_lg'):
        super().__init__(pipeline)
        self.spacy_core = spacy_core
        self._spacy_processor = spacy.load(spacy_core)
        self.STOPWORDS = self._spacy_processor.Defaults.stop_words

    def _vectorize(self, df: pd.DataFrame) -> pd.DataFrame:
        vectors_list, has_vector_list = list(), list()
        for token in df['text']:
            v = self._spacy_processor(str(token))[0]
            vectors_list.append(v.vector)
            has_vector_list.append(not v.is_oov)
        df['vector'] = vectors_list
        df['has_vector'] = has_vector_list
        return df


class GensimTokenProcessor(TokenFilteringMixin, BasePipe):
    """
    Class for tokenization using gensim
    """

    def __init__(self, pipeline: List[str], gensim_model_path: str):
        super().__init__(pipeline)
        self.gensim_model_path = gensim_model_path
        binary = self.gensim_model_path.endswith('.bin')
        self._w2v_model = models.KeyedVectors.load_word2vec_format(self.gensim_model_path, binary=binary)
        self.STOPWORDS = STOPWORDS

    def _vectorize(self, df: pd.DataFrame) -> pd.DataFrame:
        vectors_list, has_vector_list = list(), list()
        for token in df['text']:
            try:
                vector = self._w2v_model[str(token)]
                vectors_list.append(vector)
                has_vector_list.append(True)
            except KeyError:
                vectors_list.append(np.array([0] * 300, dtype=np.float32))
                has_vector_list.append(False)
        df['vector'] = vectors_list
        df['has_vector'] = has_vector_list
        return df

    def vectorize(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._vectorize(df)
