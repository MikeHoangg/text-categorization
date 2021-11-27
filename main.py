import logging

from src.base import ProductTextProcessor
from src.utils import export_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)

if __name__ == '__main__':
    dataset_file = 'data/dataset/shoes_title_description_dataset.csv'

    config_file = 'data/gensim/config.yaml'
    output = 'data/gensim/cores.json'

    processor = ProductTextProcessor(config_file, dataset_file)
    res = processor.run()
    export_json(res, output)
