#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}
PORT=$(rand 1024 2048)
timestamp=$(date +%Y%m%d_%H%M%S)

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --distribute ${@:3} --timestamp $timestamp
