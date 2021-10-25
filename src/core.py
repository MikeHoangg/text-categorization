import itertools

import spacy
import pandas as pd

from modulefinder import Module

from config import load_config


class Processor:
    def __init__(self, config_path: str, dataset_path: str):
        config = load_config(config_path)
        self.process_field = config['process_field']

        self.spacy_core = config['spacy_core']
        self._spacy_processor = spacy.load(self.spacy_core)

        self.preprocess_pipeline = config['config']['preprocess']
        self.process_pipeline = config['config']['process']
        self.pipeline = config['config']['pipeline']

        # todo: add dataset path validation
        self.dataset = pd.read_json(dataset_path)

    def run_pipeline(self, module: Module, pipeline: list, df: pd.DataFrame) -> dict:
        """
        Method that runs a pipeline, using module functions
        """
        for pipe in pipeline:
            pipe_func = getattr(module, pipe)
            df = pipe_func(df)
        return df

    def get_tokens(self, df: pd.DataFrame):
        """
        Method for getting tokens from DataFrame
        """
        return list(itertools.chain.from_iterable([self._spacy_processor(title) for title in df[self.process_field]]))
