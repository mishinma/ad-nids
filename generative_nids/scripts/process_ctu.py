
import logging

from generative_nids.process.argparser import get_argparser
from generative_nids.process.ctu import create_ctu_dataset, process_ctu_data

ORIG_TRAIN_SCENARIOS = [3, 4, 5, 7, 10, 11, 12, 13]
ORIG_TEST_SCENARIOS = [1, 2, 6, 8, 9]

parser = get_argparser()
args = parser.parse_args()

loglevel = getattr(logging, args.logging.upper(), None)
logging.basicConfig(level=loglevel)

train_scenarios = ORIG_TRAIN_SCENARIOS
test_scenarios = ORIG_TEST_SCENARIOS

aggr_dir = args.aggr_dir if args.aggr_dir else args.root_dir
process_ctu_data(args.root_dir, aggr_dir, args.processes, args.frequency)

dataset = create_ctu_dataset(
    aggr_dir, train_scenarios=train_scenarios,
    test_scenarios=test_scenarios, frequency=args.frequency
)
dataset.write_to(args.out_dir, plot=False)
