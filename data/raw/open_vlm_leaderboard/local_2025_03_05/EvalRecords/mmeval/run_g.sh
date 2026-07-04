#!/bin/bash
set -x
torchrun --nproc-per-node=$1 run.py ${@:2}
