import itertools
import os
import argparse
import re

import pandas as pd
import numpy as np
import multiprocessing as mp

from datetime import datetime
from nltk.tokenize import word_tokenize

from main import run, export, OUTPUT_FOLDER

from scripts import PUNCTUATION_TABLE, STOP_WORDS, SPACY_PROCESSOR

CONSONANTS = re.compile(r'^[bcdfghjklmnpqrstvwxyz]+$')
VOWELS = re.compile(r'^[aeiou]+$')


def nltk_process_text(text: str) -> list:
    """
    Function for processing text with nltk library
    :param text: text for processing
    :return: list of word tokens
    """
    tokens = word_tokenize(text)
    # clear punctuation
    tokens = [w.translate(PUNCTUATION_TABLE) for w in tokens]
    # clear not alphabetic
    tokens = [w for w in tokens if w.isalpha()]
    # clear stop words
    tokens = [w for w in tokens if w not in STOP_WORDS]
    # clear single characters
    tokens = [w for w in tokens if len(w) > 1]
    # clear words that consist only from consonants
    tokens = [w for w in tokens if not CONSONANTS.match(w)]
    # clear words that consist only from vows
    tokens = [w for w in tokens if not VOWELS.match(w)]
    return tokens


def spacy_process_text(text: str) -> list:
    """
    Function for processing text with spacy library
    :param text: text for processing
    :return: list of word tokens
    """
    return list(
        map(
            lambda y: y.text,
            # filter out stop words, not alphabetic symbols, without vector, single characters
            filter(
                lambda x:
                # clear not alphabetic
                x.is_alpha
                # clear stop words
                and not x.is_stop
                # clear tokens without vector
                and not x.is_oov
                # clear single characters
                and len(x.text) > 1
                # clear words that consist only from consonants
                and not CONSONANTS.match(x.text)
                # clear words that consist only from vows
                and not VOWELS.match(x.text),
                SPACY_PROCESSOR(text)
            )
        )
    )


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function for preprocessing DataFrame
    :param data: DataFrame obj
    :return: preprocessed DataFrame
    """
    # cleaning DataFrame from empty texts
    data['title'].replace('', np.nan, inplace=True)
    data.dropna(subset=['title'], inplace=True)
    # lowercase all data
    data['title'] = data['title'].str.lower()
    # dropping duplicate text
    return data.drop_duplicates(subset=['title'])


def process_data(data: pd.DataFrame) -> list:
    """
    Function for processing DataFrame
    :param data: DataFrame obj
    :return: list of word tokens
    """
    print(f"DataFrame size: {len(data.index)} rows, {len(data.columns)} cols")

    print(f'processing started: {datetime.now()}')
    # words = [nltk_process_text(title) for title in data['title']]
    words = [spacy_process_text(title) for title in data['title']]
    words = list(itertools.chain.from_iterable(words))
    print(f'processing finished: {datetime.now()}')

    return words


def multi_process_data(data: pd.DataFrame) -> list:
    """
    Function for processing DataFrame using multiprocessing
    :param data: DataFrame obj
    :return: list of word tokens
    """
    with mp.Pool(mp.cpu_count()) as pool:
        words = pool.map(process_data, np.array_split(data, mp.cpu_count()))
        words = list(itertools.chain.from_iterable(words))
        return words


def create_word_count(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function for counting words in each text
    :param data: DataFrame obj
    :return: DataFrame with word count
    """
    data = preprocess_data(data)

    # words = process_data(data)
    words = multi_process_data(data)

    print(f'started counting: {datetime.now()}, unique words len {len(set(words))}, words len {len(words)}')
    count_series = pd.Series(words).value_counts()
    print(f'finished counting: {datetime.now()}')

    return pd.DataFrame({'word': count_series.index, 'count': count_series.values})


@run
def main():
    def read_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', '-i', help='Full path for input file', type=str, required=True)
        parser.add_argument('--output_path', '-o', help=f'Name for output file, it will be created in {OUTPUT_FOLDER}',
                            type=str, required=True)
        return parser.parse_args()

    args = read_args()
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    if args.input_path.find('.json') < 0:
        raise Exception('input file must have json extension')

    data = pd.read_json(args.input_path)
    res = create_word_count(data)
    export(res, os.path.join(OUTPUT_FOLDER, args.output_path))


if __name__ == '__main__':
    main()
