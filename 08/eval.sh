#!/bin/bash

PREDICTIONS=$1

python /home/liefe/py/hw/08/morpho_eval.py --system $PREDICTIONS --gold "/home/liefe/py/hw/08/czech-pdt-test.txt"
