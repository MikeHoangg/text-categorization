import os
import argparse

import pandas as pd

from datetime import datetime

from main import run, export, OUTPUT_FOLDER
from scripts import SPACY_PROCESSOR


def create_core(data: list, percent_threshold: float, word_threshold: int = None) -> pd.DataFrame:
    """
    Function for comparing words
    :param data: list of words
    :param percent_threshold: threshold for filtering out words
    :param word_threshold: threshold for selecting top words
    :return: core DataFrame
    """
    print(f'percent threshold - {percent_threshold}%; word threshold - {word_threshold}')
    if word_threshold is not None:
        data = data[:word_threshold]
    words = list(SPACY_PROCESSOR(' '.join(data)))
    res_data = []

    # comparing word pairs
    print(f'started comparing: {datetime.now()}')
    percent_threshold = percent_threshold / 100
    for word_1 in words:
        for word_2 in words:
            if word_1.similarity(word_2) < percent_threshold:
                break
        else:
            res_data.append(word_1.text)
    print(f'finished comparing: {datetime.now()}')

    res = pd.DataFrame({'word': res_data})
    return res


@run
def main():
    def read_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--word_threshold', '-w', help='Threshold for comparison', type=int, required=False)
        parser.add_argument('--percent_threshold', '-p', help='Threshold for comparison', type=float, required=False,
                            default=50)
        parser.add_argument('--input_path', '-i', help='Full path for input file', type=str, required=True)
        parser.add_argument('--output_path', '-o', help=f'Name for output file, it will be created in {OUTPUT_FOLDER}',
                            type=str, required=True)
        return parser.parse_args()

    args = read_args()
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    if args.input_path.find('.json') < 0:
        raise Exception('input file must have json extension')

    clusters = pd.read_json(args.input_path)
    for idx, cluster in enumerate(clusters['cluster']):
        res = create_core(cluster, args.percent_threshold, args.word_threshold)
        export(res, os.path.join(OUTPUT_FOLDER, f'{str(idx)}_{args.output_path}'))


if __name__ == '__main__':
    main()
