import sys
import requests
import logging

import click
from ml_classifier.data.dataset import read_data
from ml_classifier_online.entities import read_requests_params, Sample


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command()
@click.argument('config-path', default='configs/requests_config.yaml')
def run_request(config_path: str) -> None:
    logger.info(f'Start with config: {config_path}')
    requests_params = read_requests_params(config_path)
    logger.info(f'Request params: {requests_params}')
    data = read_data(requests_params.path_to_data).to_dict('records')
    logger.info(f'Data size: {len(data)}')
    url = f'http://{requests_params.host}:{requests_params.port}/predict/'
    logger.info('Start predicting...')
    for i in range(len(data)):
        sample = Sample(**data[i]).dict()
        response = requests.post(url=url, json=sample)
        status = response.status_code
        predict = response.json()
        logger.info(f'sample {i}: status_code: {status}, predict: {predict}')

    logger.info('end')


if __name__ == '__main__':
    run_request()



