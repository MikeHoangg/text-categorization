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
from typing import Any
from spacy.tokens import Token

from .config import load_config
from .data_processing import preprocessing, token_processing, processing


class ProductTextProcessor:
    """
    Class for processing DataFrame of product offers
    With defined pipeline it runs functions from builtin modules for processing
    """

    def __init__(self, config_path: str, dataset_path: str):
        config = load_config(config_path)
        self.process_field = config['process_field']

        self.spacy_core = config['spacy_core']
        self._spacy_processor = spacy.load(self.spacy_core)

        self.preprocess_pipeline = config['pipeline']['preprocess']
        self.token_process_pipeline = config['pipeline']['token_process']
        self.process_pipeline = config['pipeline']['process']

        # todo: add dataset path validation
        self.dataset = pd.read_json(dataset_path)

    def run_pipeline(self, module: builtins, pipeline: list, *args, **kwargs) -> Any:
        """
        Method that runs a pipeline, using module functions
        """
        res = None
        for pipe in pipeline:
            pipe_func = getattr(module, pipe)
            res = pipe_func(*args, **kwargs)
        return res

    # todo: doesn't run pipeline, fix it
    def run(self) -> pd.DataFrame:
        """
        Method for running pipeline
        """
        df = self._data_preprocessing(df=self.dataset, column=self.process_field)
        tokens = self._tokens_processing(df)
        return self._data_processing(tokens=tokens)

    def _data_preprocessing(self, *args, **kwargs) -> pd.DataFrame:
        """
        Method for preprocessing DataFrame
        """
        return self.run_pipeline(preprocessing, self.preprocess_pipeline, *args, **kwargs)

    def token_modifier(self, token: Token) -> dict:
        """
        Method for modifying spacy tokens into basic type
        """
        return {
            'text': token.text,
            'is_oov': token.is_oov,
            'is_stop': token.is_stop,
            'is_alpha': token.is_alpha,
            'vector': token.vector,
        }

    def _get_tokens(self, df: pd.DataFrame) -> list:
        """
        Method for getting tokens from DataFrame
        """
        tokens = list(itertools.chain.from_iterable([self._spacy_processor(title) for title in df[self.process_field]]))
        tokens = [self.token_modifier(token) for token in tokens]
        return self.run_pipeline(token_processing, self.token_process_pipeline, tokens=tokens)

    # todo: change structure to pd.DataFrame instead of list
    def _tokens_processing(self, df: pd.DataFrame) -> list:
        """
        Method for getting tokens from DataFrame
        """
        with Pool() as pool:
            tokens = pool.map(self._get_tokens, np.array_split(df, os.cpu_count()))
            tokens = list(itertools.chain.from_iterable(tokens))
            return tokens

    def _data_processing(self, *args, **kwargs) -> Any:
        return self.run_pipeline(processing, self.process_pipeline, *args, **kwargs)
