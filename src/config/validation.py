"""
Module for configuration file validation
"""
import trafaret as t

from ..data_processing import preprocessing, processing, token_processing
from ..errors import INVALID_SPACY_CORE, INVALID_FUNCTION


class SpacyCoreString(t.String):
    # MARK: can be updated for different core usages
    AVAILABLE_SPACY_CORES = {
        'en_core_web_lg'
    }

    def check_and_return(self, value):
        value = super().check_and_return(value)
        if value not in self.AVAILABLE_SPACY_CORES:
            self._failure(
                f'no such spacy core is available',
                value=value,
                code=INVALID_SPACY_CORE
            )
        return value


class ModuleAttrString(t.String):
    def __init__(self, module, **kwargs):
        self.__module = module
        super().__init__(**kwargs)

    def check_and_return(self, value):
        value = super().check_and_return(value)
        if not hasattr(self.__module, value):
            self._failure(
                f'no such function available in {self.__module.__name__}',
                value=value,
                code=INVALID_FUNCTION
            )
        return value


# MARK: main configurator
CONFIG_FILE_VALIDATOR = t.Dict(
    {
        t.Key('spacy_core'): SpacyCoreString,
        t.Key('pipeline'): t.Dict(
            {
                t.Key('preprocess'): t.Dict({
                    t.Key('pipes'): t.List(ModuleAttrString(preprocessing)),
                    t.Key('args', optional=True): t.Dict(allow_extra='*')
                }),
                t.Key('token_process'): t.Dict({
                    t.Key('pipes'): t.List(ModuleAttrString(token_processing)),
                    t.Key('args', optional=True): t.Dict(allow_extra='*')
                }),
                t.Key('process'): t.Dict({
                    t.Key('pipes'): t.List(ModuleAttrString(processing)),
                    t.Key('args', optional=True): t.Dict(allow_extra='*')
                })
            }
        )
    }
)
