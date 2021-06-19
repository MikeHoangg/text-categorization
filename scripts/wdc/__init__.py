import os
import spacy

from main import OUTPUT_FOLDER

RES_FOLDER = os.path.join(OUTPUT_FOLDER, 'wdc')
EN_PIPELINE = 'en_core_web_sm'
NLP = spacy.load(EN_PIPELINE)
