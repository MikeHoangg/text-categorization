"""
Main module for processing data classes
"""
import itertools
import os
import builtins

import spacy
import pandas as pd
import numpy as np

from multiprocessing import Pool
from spacy.tokens import Token

from .config import load_config
from .data_processing import preprocessing, token_processing, processing


class ProductTextProcessor:
    """
    Class for processing DataFrame of product offers
    With defined pipeline it runs functions from builtin modules for processing
    """
    _dataset: pd.DataFrame = ...

    def __init__(self, config_path: str, dataset_path: str):
        config = load_config(config_path)

        # loading spacy processor
        self.spacy_core = config['spacy_core']
        self._spacy_processor = spacy.load(self.spacy_core)

        # assigning pipelines
        self.preprocess_pipeline = config['pipeline']['preprocess']
        self.token_process_pipeline = config['pipeline']['token_process']
        self.process_pipeline = config['pipeline']['process']

        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f'{dataset_path} - no such file.')
        self._dataset = pd.read_json(dataset_path)

    @staticmethod
    def run_pipeline(module: builtins, pipeline: list, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Method that runs a pipeline, using module functions
        """
        for pipe in pipeline:
            pipe_func = getattr(module, pipe)
            df = pipe_func(df, *args, **kwargs)
        return df

    def run(self) -> pd.DataFrame:
        """
        Method for running pipeline
        """
        preprocessed_df = self._data_preprocessing(self._dataset)
        tokens_df = self._tokens_processing(preprocessed_df)
        return self._data_processing(tokens_df)

    def _data_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for preprocessing DataFrame
        """
        return self.run_pipeline(preprocessing, self.preprocess_pipeline['pipes'], df,
                                 **self.preprocess_pipeline.get('args', {}))

    def __token_modifier(self, token: Token) -> dict:
        """
        Method for modifying spacy tokens into basic type
        """
        return {
            'text': token.text,
            'is_oov': token.is_oov,
            'is_stop': token.is_stop,
            'is_alpha': token.is_alpha,
            'vector': token.vector,
            'vector_norm': token.vector_norm,
        }

    def _get_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for getting tokens from DataFrame
        """
        tokens = list(itertools.chain.from_iterable([self._spacy_processor(title) for title in df[self.process_field]]))
        tokens = [self.__token_modifier(token) for token in tokens]
        return pd.DataFrame.from_records(tokens)

    def _tokens_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for getting tokens from DataFrame
        """
        with Pool() as pool:
            token_dataframes = pool.map(self._get_tokens, np.array_split(df, os.cpu_count()))
            df = pd.concat(token_dataframes)
        return self.run_pipeline(token_processing, self.token_process_pipeline['pipes'], df,
                                 **self.token_process_pipeline.get('args', {}))

    def _data_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.run_pipeline(processing, self.process_pipeline['pipes'], df,
                                 **self.process_pipeline.get('args', {}))
