import os
import argparse
import pandas as pd

from datetime import datetime
from fuzzywuzzy import fuzz

from main import OUTPUT_FOLDER, run, export


def compare_words(path: str, out_path: str, percent_threshold: int):
    data = pd.read_json(path)
    words = [word for word in data.word]
    res = []
    print(f'started comparing: {datetime.now()}')
    while words:
        word = words.pop(0)
        for w in words:
            ratio = fuzz.ratio(word, w)
            if ratio >= percent_threshold:
                res.append((word, w, ratio))
        # TODO remove
        print(len(words))
    res = pd.DataFrame(res, columns=['word1', 'word2', 'percent'])
    export(res, os.path.join(OUTPUT_FOLDER, out_path))


@run
def main():
    def read_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--percent_threshold', '-p', help='Threshold for comparison', type=int, required=True)
        parser.add_argument('--input_path', '-i', help='Full path for input file', type=str, required=True)
        parser.add_argument('--output_path', '-o', help=f'Name for output file, it will be created in {OUTPUT_FOLDER}',
                            type=str, required=True)
        return parser.parse_args()

    args = read_args()
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    if args.input_path.find('.json') < 0:
        raise Exception('input file must have json extension')
    compare_words(args.input_path, args.output_path, args.percent_threshold)


if __name__ == '__main__':
    main()
