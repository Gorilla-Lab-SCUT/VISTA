#!/bin/bash

TASK_DESC=$1
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=/data/Outputs/VISTA
CONFIG=$2
NUSC_CBGS_WORK_DIR=$OUT_DIR/VISTA_$TASK_DESC\_$DATE_WITH_TIME
if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port 37447 ./tools/train.py $CONFIG --work_dir=$NUSC_CBGS_WORK_DIR