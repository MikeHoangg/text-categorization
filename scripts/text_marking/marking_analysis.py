import re
import os
import argparse
import json
from typing import Union

from box import Box
from main import OUTPUT_FOLDER, run

RES_FOLDER = os.path.join(OUTPUT_FOLDER, 'med_texts')
TAG = re.compile(r'<complex>(.*?)</complex>')
EXCLUDE = re.compile(r'^readme.*$')
INCLUDE = re.compile(r'^.*\.txt$')
DEPTH = 2


def collect_statistics(path: str, out_file_name: str):
    count_res = Box(box_dots=True, default_box=True)

    for base, _, files in os.walk(path):
        os.path.dirname(base)
        for file in files:
            if INCLUDE.match(file) and not EXCLUDE.match(file):
                with open(os.path.join(base, file)) as f:
                    text = ''.join(f.readlines())
                    words = re.findall(TAG, text)
                    for word in words:
                        count_res.get(word)

    for base, _, files in os.walk(path):
        os.path.dirname(base)
        for file in files:
            if INCLUDE.match(file) and not EXCLUDE.match(file):
                with open(os.path.join(base, file)) as f:
                    text = ''.join(f.readlines())
                    for word in count_res.keys():
                        box = count_res[word]
                        folders = base.split(os.sep)
                        for idx, folder in enumerate(folders):
                            if idx + DEPTH >= len(folders):
                                box = box[folder]
                        name = file[:-4]

                        count = len(re.findall(re.escape(word), text, re.IGNORECASE | re.UNICODE))
                        if name in box:
                            box[name] += count
                        else:
                            box[name] = count

    total_count_res = {k: get_counts(v) for k, v in count_res.items()}
    with open(os.path.join(RES_FOLDER, out_file_name), 'w') as write_file:
        json.dump(count_res, write_file)
    with open(os.path.join(RES_FOLDER, f'total_{out_file_name}'), 'w') as write_file:
        json.dump(total_count_res, write_file)


def get_counts(value: Union[dict, int]) -> int:
    if isinstance(value, dict):
        return sum([get_counts(value[n]) for n in value.keys()])
    else:
        return value


@run
def main():
    def read_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', '-i', help='Full path for input folder', type=str,
                            required=True)
        parser.add_argument('--output_path', '-o',
                            help=f'Name for output file, it will be created in {RES_FOLDER}',
                            type=str, required=True)
        return parser.parse_args()

    args = read_args()
    if not os.path.exists(RES_FOLDER):
        os.mkdir(RES_FOLDER)
    if not os.path.isdir(args.input_path):
        raise Exception('input_path must be a folder')

    collect_statistics(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
