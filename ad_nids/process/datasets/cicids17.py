import shutil
import os
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


def download_cicids17(dataset_path):

    logging.info('Downloading the dataset')

    dataset_path = Path(dataset_path).resolve()
    dataset_path.mkdir(parents=True)

    mycwd = os.getcwd()
    os.chdir(dataset_path)

    try:
        check_output(
            'wget --header="Host: 205.174.165.80" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36" --header="Accept:'
            ' text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/'
            'signed-exchange;v=b3" --header="Accept-Language: en-US,en;q=0.9" --header="Referer:'
            ' http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/" "http://205.174.165.80/CICDataset/CIC-IDS-2017/'
            'Dataset/GeneratedLabelledFlows.zip" -O "GeneratedLabelledFlows.zip" -c', shell=True
        )
    except Exception as e:
        raise DownloadError('Could not download the dataset')

    check_output(['unzip', "GeneratedLabelledFlows.zip"])
    os.remove(dataset_path/"GeneratedLabelledFlows.zip")

    middle_path = list(dataset_path.iterdir())[0]  # 'TrafficLabelling '
    for path in middle_path.iterdir():
        shutil.move(path, dataset_path/path.name)
    shutil.rmtree(middle_path)

    os.chdir(mycwd)  # go back where you came from

    return
