#!/bin/bash

# add any arg for verbose output...
if [ $# -gt 0 ]; then
  export VERBOSE=True && coverage run run_tests.py
else
#  coverage run run_tests.py --pattern test_nine
  coverage run run_tests.py
fi

coverage report -m
