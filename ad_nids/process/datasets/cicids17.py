import shutil
import logging

from datetime import datetime
from subprocess import CalledProcessError, check_output
from pathlib import Path

import pandas as pd
import numpy as np


from ad_nids.utils.exception import DownloadError
from ad_nids.dataset import Dataset, create_meta
from ad_nids.utils.misc import sample_df, dd_mm_yyyy2mmdd
from ad_nids.process.columns import CIC_IDS_ORIG_COLUMNS, CIC_IDS_ATTACK_LABELS


DATASET_NAME = 'CIC-IDS2017'


def download_cicids(dataset_path):

    logging.info('Downloading the dataset')

    dataset_path = Path(dataset_path)
    dataset_path.mkdir(parents=True)

    try:
        check_output(
            ['aws', 's3', 'sync',
             '--region', 'eu-north-1',
             '--no-sign-request', "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms",
             dataset_path])
        returncode = 0
    except CalledProcessError as e:
        returncode = e.returncode

    if returncode != 0:
        raise DownloadError('Could not download the dataset')

    return
