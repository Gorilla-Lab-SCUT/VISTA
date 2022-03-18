#!/bin/bash

CONFIG=$1
WORK_DIR=$2
CHECKPOINT=$3

# Test
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --master_port 7988 \
    --nproc_per_node=4 \
    ./tools/dist_test.py \
    $CONFIG \
    --work_dir=$WORK_DIR \
    --checkpoint=$CHECKPOINT \
    # --test True
