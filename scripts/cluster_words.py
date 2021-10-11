import os
import argparse
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

from scripts import SPACY_PROCESSOR
from main import run, OUTPUT_FOLDER, export


def cluster_words(data: pd.DataFrame, num_of_clusters: int = 20) -> pd.DataFrame:
    """
    Function for clustering words into groups
    :param data: DataFrame obj
    :param num_of_clusters: number of clusters
    :return: DataFrame of clusters
    """
    data = SPACY_PROCESSOR(' '.join(data['word']))
    data = pd.DataFrame([(x.text, x.vector) for x in data], columns=['word', 'vector'])
    kmeans_words = KMeans(n_clusters=num_of_clusters).fit(np.array([x for x in data['vector']]))
    data['labels'] = kmeans_words.labels_
    res = [names['word'].tolist() for label, names in data.groupby('labels')]
    return pd.DataFrame([(cluster,) for cluster in res], columns=['cluster'])


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

    data = pd.read_json(args.input_path)
    res = cluster_words(data)
    export(res, os.path.join(OUTPUT_FOLDER, args.output_path))


if __name__ == '__main__':
    main()
