import logging

from src.base import ProductTextProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)

if __name__ == '__main__':
    # config_file = 'config.yaml'
    config_file = 'dumps/spacy_config.yaml'
    dataset_file = 'dumps/shoes_title_dataset.json'

    processor = ProductTextProcessor(config_file, dataset_file)
    res = processor.run()
    pass
