#!/usr/bin/env bash

# check the enviroment info

PYTHON="/usr/bin/python3"
export PYTHONPATH="/data/liangzhiyuan/projects/TEL":$PYTHONPATH

cd ../../../
${PYTHON} lib/metrics/cityscapes/setup.py build_ext --inplace
