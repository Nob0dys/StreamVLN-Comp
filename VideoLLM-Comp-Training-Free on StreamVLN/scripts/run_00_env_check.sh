#!/bin/zsh
set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export PYTHONPATH=/home/ubuntu/project/StreamVLN:$PYTHONPATH
cd /home/ubuntu/project/StreamVLN

echo "[run_00] Environment check"
python test_train_setup.py
python test_training_simple.py

echo "[run_00] Done"
