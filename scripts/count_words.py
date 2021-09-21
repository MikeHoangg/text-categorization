import os
import argparse

import pandas as pd
import numpy as np

from datetime import datetime

from main import run, export, OUTPUT_FOLDER
from nltk.tokenize import word_tokenize

from scripts import PUNCTUATION_TABLE, STOP_WORDS, STEMMER, SPACY_STOP_WORDS


def count_words(path: str, out_path: str) -> pd.DataFrame:
    """
    Function for counting words in each text
    :param path: path of source file
    :param out_path: path of result file
    :return:
    """
    data = pd.read_json(path)
    words = []

    # cleaning DataFrame from empty texts
    data['title'].replace('', np.nan, inplace=True)
    data.dropna(subset=['title'], inplace=True)

    for title in data['title']:
        tokens = word_tokenize(title)
        # lowercase all data
        tokens = [w.lower() for w in tokens]
        # clear punctuation
        tokens = [w.translate(PUNCTUATION_TABLE) for w in tokens]
        # clear not alphabetic
        tokens = [w for w in tokens if w.isalpha()]
        # clear stop words
        tokens = [w for w in tokens if w not in STOP_WORDS and w not in SPACY_STOP_WORDS]
        # stemming of words
        tokens = [STEMMER.stem(w) for w in tokens]
        words += tokens

    print(f'started counting: {datetime.now()}, unique words len {len(set(words))}, words len {len(words)}')
    count_series = pd.Series(words).value_counts()
    print(f'finished counting: {datetime.now()}')

    res = pd.DataFrame({'word': count_series.index, 'count': count_series.values})
    export(res, os.path.join(OUTPUT_FOLDER, out_path))
    return res


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
    count_words(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
