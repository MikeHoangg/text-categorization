import os
import argparse
import pandas as pd
import multiprocessing as mp
import datetime
import uuid

from main import OUTPUT_FOLDER, run, export

RES_FOLDER = os.path.join(OUTPUT_FOLDER, 'wdc')


def process_chunk(chunk: pd.DataFrame, query: str) -> tuple:
    res = chunk.query(query)
    return res, res.filter(items=['id', 'title'])


def create_file_with_category(path: str, compression: str, query: str, out_file_name: str, chunk_size: int = 10 ** 6):
    funcs = []
    opts = {
        'lines': True,
        'chunksize': chunk_size
    }
    if compression:
        opts['compression'] = compression
    with mp.Pool(mp.cpu_count()) as pool, pd.read_json(path, **opts) as reader:
        for idx, chunk in enumerate(reader, 1):
            print(f'processing chunk #{idx}: {datetime.datetime.now()}')
            funcs.append(pool.apply_async(process_chunk, [chunk, query]))

    print(f'concatenating chunks')
    res = tuple(zip(func.get() for func in funcs))
    export(pd.concat(res[0]), os.path.join(RES_FOLDER, out_file_name))
    export(pd.concat(res[1]), os.path.join(RES_FOLDER, f'id_title_{out_file_name}'))


def create_chunk_files_with_category(path: str, compression: str, query: str, out_file_name: str,
                                     chunk_size: int = 10 ** 6):
    opts = {
        'lines': True,
        'chunksize': chunk_size
    }
    if compression:
        opts['compression'] = compression
    with mp.Pool(mp.cpu_count()) as pool, pd.read_json(path, **opts) as reader:
        for idx, chunk in enumerate(reader, 1):
            print(f'processing chunk #{idx}: {datetime.datetime.now()}')
            func = pool.apply_async(process_chunk, [chunk, query])
            res = func.get()
            _id = uuid.uuid4()

            export(res[0], os.path.join(RES_FOLDER, f'{_id}_{out_file_name}'))
            export(res[1], os.path.join(RES_FOLDER, f'{_id}_id_title_{out_file_name}'))


@run
def main():
    def read_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', '-i', help='Full path for input file', type=str, required=True)
        parser.add_argument('--output_path', '-o', help=f'Name for output file, it will be created in {RES_FOLDER}',
                            type=str, required=True)
        parser.add_argument('--category', '-cat', help='Category to filter', type=str, required=True)
        parser.add_argument('--compression', '-comp', help='Input file compression', type=str)
        return parser.parse_args()

    args = read_args()
    if not os.path.exists(RES_FOLDER):
        os.mkdir(RES_FOLDER)
    if args.input_path.find('.json') < 0:
        raise Exception('input file must have json extension')

    print(f'processing file - {args.input_path}, category - {args.category}')
    # create_file_with_category(args.input_path, args.compression, f'category == "{args.category}"', args.output_path)
    create_chunk_files_with_category(args.input_path, args.compression, f'category == "{args.category}"',
                                     args.output_path)


if __name__ == '__main__':
    main()
