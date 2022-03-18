#!/bin/bash

TASK_DESC=$1
RESUME=$3
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

# Voxelnet
if [ ! $RESUME ]
then
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 37447 ./tools/train.py $CONFIG --work_dir=$NUSC_CBGS_WORK_DIR
else
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py $CONFIG --work_dir=$NUSC_CBGS_WORK_DIR --resume_from=$RESUME
fi