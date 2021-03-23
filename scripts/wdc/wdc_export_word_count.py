import os
import argparse
import pandas as pd
import numpy as np

from datetime import datetime

from main import OUTPUT_FOLDER, run


def count_words(path: str, out_path: str):
    data = pd.read_json(path)
    data['title'].replace('', np.nan, inplace=True)
    data.dropna(subset=['title'], inplace=True)
    words = []
    for title in data['title']:
        words += title.split()
    print(f'started counting: {datetime.now()}')
    res = pd.DataFrame([(word, words.count(word)) for word in set(words)], columns=['word', 'count'])
    res = res.sort_values(by='count', ascending=False)
    print(f'dataframe has {len(res)} rows')
    file_path = os.path.join(OUTPUT_FOLDER, out_path)
    print(f'started exporting: {datetime.now()}')
    res.to_json(file_path)
    print(f'created {file_path}: {datetime.now()}')


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
