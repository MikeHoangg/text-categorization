import os
import argparse
import pandas as pd
import numpy as np
import spacy
import re

from datetime import datetime

from main import OUTPUT_FOLDER, run

RES_FOLDER = os.path.join(OUTPUT_FOLDER, 'wdc')
NLP = spacy.load("en_core_web_sm")
IS_WORD = re.compile(r'^[a-zA-Z]{3,}$')


def count_words(path: str, out_path: str, count_threshold: int = 50):
    data = pd.read_json(path)
    data['title'].replace('', np.nan, inplace=True)
    data.dropna(subset=['title'], inplace=True)
    words = []
    for title in data['title']:
        words += list(
            map(
                lambda y: y.text,
                filter(
                    lambda x: not x.is_stop and x.is_alpha and IS_WORD.match(x.text),
                    NLP(title)
                )
            )
        )
    unique_words = set(words)
    print(f'started counting: {datetime.now()}, words len {len(unique_words)}')
    res_data = []
    for word in unique_words:
        if word_count := words.count(word):
            if word_count >= count_threshold:
                res_data.append((word, word_count))
    res = pd.DataFrame(res_data, columns=['word', 'count'])
    res = res.sort_values(by='count', ascending=False)
    print(f'dataframe has {len(res)} rows')
    file_path = os.path.join(RES_FOLDER, out_path)
    print(f'started exporting: {datetime.now()}')
    res.to_json(file_path)
    print(f'created {file_path}: {datetime.now()}')


@run
def main():
    def read_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', '-i', help='Full path for input file', type=str, required=True)
        parser.add_argument('--output_path', '-o', help=f'Name for output file, it will be created in {RES_FOLDER}',
                            type=str, required=True)
        return parser.parse_args()

    args = read_args()
    if not os.path.exists(RES_FOLDER):
        os.mkdir(RES_FOLDER)
    if args.input_path.find('.json') < 0:
        raise Exception('input file must have json extension')
    count_words(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
