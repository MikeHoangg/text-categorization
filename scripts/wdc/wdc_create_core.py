import os
import argparse
import pandas as pd
import spacy

from datetime import datetime

from main import OUTPUT_FOLDER, run, export

RES_FOLDER = os.path.join(OUTPUT_FOLDER, 'wdc')
NLP = spacy.load("en_core_web_sm")


def compare_words(path: str, out_path: str, percent_threshold: float):
    print(f'threshold {percent_threshold}%')
    out_path = os.path.join(RES_FOLDER, f'{percent_threshold}_{out_path}')
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
    export(res, out_path)


@run
def main():
    def read_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--percent_threshold', '-p', help='Threshold for comparison', type=float, required=True)
        parser.add_argument('--input_path', '-i', help='Full path for input file', type=str, required=True)
        parser.add_argument('--output_path', '-o', help=f'Name for output file, it will be created in {RES_FOLDER}',
                            type=str, required=True)
        return parser.parse_args()

    args = read_args()
    if not os.path.exists(RES_FOLDER):
        os.mkdir(RES_FOLDER)
    if args.input_path.find('.json') < 0:
        raise Exception('input file must have json extension')
    compare_words(args.input_path, args.output_path, 1.5)


if __name__ == '__main__':
    main()
