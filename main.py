import logging

from src.base import ProductTextProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)

if __name__ == '__main__':
    config_file = '/home/mikehoang/projects/khpi/text-categorization/config.yaml'
    dataset_file = '/home/mikehoang/projects/khpi/text-categorization/dumps/shoes_title_dataset.json'

    processor = ProductTextProcessor(config_file, dataset_file)
    res = processor.run()
    pass
