"""
Main module for processing data classes
"""
import os
import builtins
import yaml
import trafaret as t
import pandas as pd

from sklearn.pipeline import Pipeline

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
        except t.DataError as ex:
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
            t.Key('pipeline'): t.Dict(
                {
                    t.Key('preprocess'): t.Dict({
                        t.Key('processor'): validation.ModuleAttrString(preprocessing),
                        t.Key('pipes'): t.List(t.String),
                        t.Key('args'): t.Dict(
                            allow_extra='*'
                        )
                    }),
                    t.Key('token_process'): t.Dict({
                        t.Key('processor'): validation.ModuleAttrString(token_processing),
                        t.Key('pipes'): t.List(t.String),
                        t.Key('args', optional=True): t.Dict(
                            allow_extra='*'
                        )
                    }),
                    t.Key('process'): t.Dict({
                        t.Key('processor'): validation.ModuleAttrString(processing),
                        t.Key('pipes'): t.List(t.String),
                        t.Key('args', optional=True): t.Dict(
                            allow_extra='*'
                        )
                    })
                }
            )
        }
    )

    @property
    def preprocess_pipeline(self):
        return self.config['pipeline']['preprocess']

    @property
    def token_process_pipeline(self):
        return self.config['pipeline']['token_process']

    @property
    def process_pipeline(self):
        return self.config['pipeline']['process']

    @property
    def preprocessor(self):
        return getattr(preprocessing, self.config['pipeline']['preprocess']['processor'])

    @property
    def token_processor(self):
        return getattr(token_processing, self.config['pipeline']['token_process']['processor'])

    @property
    def processor(self):
        return getattr(processing, self.config['pipeline']['process']['processor'])

    def run(self) -> object:
        pipeline = Pipeline([
            (
                'preprocessor',
                self.preprocessor(
                    pipeline=self.preprocess_pipeline['pipes'],
                    **self.preprocess_pipeline.get('args')
                )
            ),
            (
                'token_processor',
                self.token_processor(
                    pipeline=self.token_process_pipeline['pipes'],
                    **self.token_process_pipeline.get('args')
                )
            ),
            (
                'processor',
                self.processor(
                    pipeline=self.process_pipeline['pipes'],
                    **self.process_pipeline.get('args')
                )
            ),
        ])
        return pipeline.fit_transform(self._dataset)
