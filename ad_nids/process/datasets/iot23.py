import shutil
import logging
import os

from datetime import datetime
from subprocess import CalledProcessError, check_output
from pathlib import Path

import pandas as pd
import numpy as np

from ad_nids.dataset import Dataset, create_meta
from ad_nids.utils.exception import DownloadError


DATASET_NAME = 'IOT-23'


def download_iot23(dataset_path):
    logging.info('Downloading the dataset')

    dataset_path = Path(dataset_path)
    dataset_path.mkdir(parents=True)

    mycwd = os.getcwd()
    os.chdir(dataset_path)

    try:
        check_output(
             'wget --header="Host: mcfp.felk.cvut.cz" '
             '--header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36" '
             '--header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3" '
             '--header="Accept-Language: en-US,en;q=0.9" "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz" '
             '-O "iot_23_datasets_small.tar.gz" -c --no-check-certificate', shell=True
            )
        returncode = 0
    except CalledProcessError as e:
        returncode = e.returncode
    finally:
        os.chdir(mycwd)  # go back where you came from

    if returncode != 0:
        raise DownloadError('Could not download the dataset')

    return


if __name__ == '__main__':
    dataset_path = '/home/emikmis/data/nids/IOT23-2'
    download_iot23(dataset_path)