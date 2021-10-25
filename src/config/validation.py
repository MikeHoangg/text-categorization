import trafaret as t

from ..data_processing import preprocessing, processing, core

INVALID_SPACY_CORE = 'invalid_spacy_core'
INVALID_FUNCTION = 'invalid_function'


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


CONFIG_FILE_VALIDATOR = t.Dict(
    {
        t.Key('process_field'): t.String,
        t.Key('spacy_core'): SpacyCoreString,
        t.Key('config'): t.Dict(
            {
                t.Key('preprocess'): t.List(ModuleAttrString(preprocessing)),
                t.Key('process'): t.List(ModuleAttrString(processing)),
                t.Key('pipeline'): t.List(ModuleAttrString(core))
            }
        )
    }
)
