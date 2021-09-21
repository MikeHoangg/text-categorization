import os
import argparse
import pandas as pd

from datetime import datetime

from main import run, export, OUTPUT_FOLDER
from scripts import NLP


def compare_words(path: str, out_path: str, percent_threshold: float):
    """
    Function for comparing words
    :param path: path of source file
    :param out_path: path of result file
    :param percent_threshold: threshold for filtering out words
    :return:
    """
    print(f'threshold - {percent_threshold}%')
    data = pd.read_json(path)
    words = list(NLP(' '.join(data['word'])))
    res = []
    print(f'started comparing: {datetime.now()}')
    percent_threshold = percent_threshold / 100
    for word_1 in words:
        for word_2 in words:
            if word_1.similarity(word_2) < percent_threshold:
                break
        else:
            res.append(word_1.text)
    res = pd.DataFrame(res, columns=['word'])
    out_path = os.path.join(OUTPUT_FOLDER, f'{percent_threshold}_{out_path}')
    export(res, out_path)


@run
def main():
    def read_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--percent_threshold', '-p', help='Threshold for comparison', type=float, required=True)
        parser.add_argument('--input_path', '-i', help='Full path for input file', type=str, required=True)
        parser.add_argument('--output_path', '-o', help=f'Name for output file, it will be created in {OUTPUT_FOLDER}',
                            type=str, required=True)
        return parser.parse_args()

    args = read_args()
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    if args.input_path.find('.json') < 0:
        raise Exception('input file must have json extension')
    compare_words(args.input_path, args.output_path, 1.5)


if __name__ == '__main__':
    main()
