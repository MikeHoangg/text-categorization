import spacy
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

EN_PIPELINE = 'en_core_web_lg'
# EN_PIPELINE = 'en_core_web_sm' # WARNING: doesn't have word vectors for load
SPACY_PROCESSOR = spacy.load(EN_PIPELINE)
SPACY_STOP_WORDS = SPACY_PROCESSOR.Defaults.stop_words

STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)
STEMMER = PorterStemmer()
