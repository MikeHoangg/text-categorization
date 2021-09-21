import os
import re
import argparse
import pandas as pd
import numpy as np

from datetime import datetime

from main import run, export
from . import RES_FOLDER, NLP

IS_WORD = re.compile(r'^[a-zA-Z]{3,}$')


def count_words(path: str, out_path: str, count_threshold: int = 50):
    """
    Function for counting words in each text
    :param path: path of source file
    :param out_path: path of result file
    :param count_threshold: threshold for filtering out words
    :return:
    """
    print(f'threshold - {count_threshold}%')
    data = pd.read_json(path)
    # cleaning DataFrame from empty texts
    data['title'].replace('', np.nan, inplace=True)
    data.dropna(subset=['title'], inplace=True)
    words = []
    for title in data['title']:
        words += list(
            map(lambda y: y.text,
                filter(lambda x: not x.is_stop and x.is_alpha and IS_WORD.match(x.text),
                       NLP(title))
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
    out_path = os.path.join(RES_FOLDER, out_path)
    export(res, out_path)


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
