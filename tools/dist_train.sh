#!/usr/bin/env bash

# hardcoded config
CONFIG=projects/configs/sparsedrive_small_stage2.py
GPUS=2
PORT=${PORT:-28651}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# conda env
source /root/anaconda3/etc/profile.d/conda.sh
conda activate sparsedrive

export CUDA_VISIBLE_DEVICES

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch "${@:1}"
