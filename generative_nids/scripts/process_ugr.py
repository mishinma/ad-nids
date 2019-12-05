"""
Process UGR 16 flow data and create a dataset

Must split the data by date first (use `split_ugr.py`)!

Example command
python process_ugr.py {data_path} {out_path} -p -1 -f T --overwrite --plot
"""


import logging

from generative_nids.process.argparser import get_argparser
from generative_nids.process.ugr import create_ugr_dataset, process_ugr_data


#ToDo: change from mock dates to real
TRAIN_DATES = ['2016-07-27', '2016-07-30']
TEST_DATES = ['2016-07-31']
ALL_DATES = TRAIN_DATES + TEST_DATES

parser = get_argparser()
args = parser.parse_args()

loglevel = getattr(logging, args.logging.upper(), None)
logging.basicConfig(level=loglevel)

train_dates = TRAIN_DATES
test_dates = TEST_DATES

aggr_dir = args.aggr_dir if args.aggr_dir else args.root_dir
process_ugr_data(args.root_dir, aggr_dir, processes=args.processes, frequency=args.frequency)
dataset = create_ugr_dataset(
    aggr_dir, train_dates=train_dates,
    test_dates=test_dates, frequency=args.frequency
)

dataset.write_to(args.out_dir, plot=args.plot,
                 overwrite=args.overwrite, archive=args.archive)
