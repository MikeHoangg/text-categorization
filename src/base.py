"""
Main module for processing data classes
"""
import itertools
import os
import builtins
import yaml

import trafaret as t
import spacy
import pandas as pd
import numpy as np

from multiprocessing import Pool
from trafaret import DataError

from spacy.tokens import Token

from . import errors
from .data_processing import preprocessing, token_processing, processing
from .config import validation


class BaseProcessor:
    """
    Base processor class for inheritance and running pipelines
    """
    _dataset: pd.DataFrame = ...
    config: dict = ...
    CONFIG_FILE_VALIDATOR = ...

    def __init__(self, config_path: str, dataset_path: str, *args, **kwargs):
        self.load_config(config_path)
        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f'{dataset_path} - no such file.')
        self._dataset = pd.read_json(dataset_path)

    def load_config(self, config_path: str):
        """
        Function that loads data from configuration file
        """
        with open(config_path) as config_file:
            config_data = yaml.safe_load(config_file)
        try:
            self.CONFIG_FILE_VALIDATOR.check(config_data)
            self.config = config_data
        except DataError as ex:
            raise errors.BrokenConfigException(ex.value)

    @staticmethod
    def run_pipeline(module: builtins, pipeline: list, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Method that runs a pipeline, using module functions
        """
        for pipe in pipeline:
            pipe_func = getattr(module, pipe)
            df = pipe_func(df, *args, **kwargs)
        return df

    def run(self):
        """
        Method for running pipeline
        """
        raise NotImplementedError


class ProductTextProcessor(BaseProcessor):
    """
    Class for processing DataFrame of product offers
    With defined pipeline it runs functions from builtin modules for processing
    """
    CONFIG_FILE_VALIDATOR = t.Dict(
        {
            t.Key('spacy_core'): validation.SpacyCoreString,
            t.Key('pipeline'): t.Dict(
                {
                    t.Key('preprocess'): t.Dict({
                        t.Key('pipes'): t.List(validation.ModuleAttrString(preprocessing)),
                        t.Key('args', optional=True): t.Dict(allow_extra='*')
                    }),
                    t.Key('token_process'): t.Dict({
                        t.Key('pipes'): t.List(validation.ModuleAttrString(token_processing)),
                        t.Key('args', optional=True): t.Dict(allow_extra='*')
                    }),
                    t.Key('process'): t.Dict({
                        t.Key('pipes'): t.List(validation.ModuleAttrString(processing)),
                        t.Key('args', optional=True): t.Dict(allow_extra='*')
                    })
                }
            )
        }
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # loading spacy processor
        self.spacy_core = self.config['spacy_core']
        self._spacy_processor = spacy.load(self.spacy_core)

    @property
    def preprocess_pipeline(self):
        return self.config['pipeline']['preprocess']

    @property
    def token_process_pipeline(self):
        return self.config['pipeline']['token_process']

    @property
    def process_pipeline(self):
        return self.config['pipeline']['process']

    def run(self) -> pd.DataFrame:
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
        if column := self.token_process_pipeline.get('args', {}).get('column'):
            tokens = list(itertools.chain.from_iterable([self._spacy_processor(title) for title in df[column]]))
            tokens = [self.__token_modifier(token) for token in tokens]
            return pd.DataFrame.from_records(tokens)
        raise errors.MissingArgException('token_process', 'column')

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
