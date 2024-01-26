'''
run all the tests.

Usage:
    python run_tests.py --pattern [test patterns] --verbosity [verbosity]

By default the pattern matching for tests is `test*`, so this script will
run every test in `tests/` starting with `test*`, but we may set the pattern
to, e.g., `test_data` to restrict matching to tests starting with `test_data*`.
'''
import unittest
import argparse
import time
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--pattern', default='test', type=str,
                    help='pattern base name (e.g. test_data)')
parser.add_argument('--verbosity', default=2, type=int,
                    help='test verbosity (int)')


def main(pattern, verbosity):
    run_time = int(time.time())
    logfilename = 'log_' + __file__.split('/')[-1].split('.')[0] \
        + str(run_time) + '.txt'
    logging.basicConfig(
        filename=logfilename, level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Starting...")
    LOGGER.info(__file__)

    pattern = pattern + '*.py'
    suite = unittest.TestLoader().discover('./tests/', pattern=pattern)
    unittest.TextTestRunner(verbosity=verbosity,warnings='once').run(suite)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
