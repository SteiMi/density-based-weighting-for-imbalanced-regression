#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DATASET=$1
WEIGHTED_LOSS=$2
ALPHA=$3
NUM_RUNS=$4

if [ -z "$DATASET" ] && [ -z "$WEIGHTED_LOSS" ] && [ -z "$ALPHA" ] && [ -z "$NUM_RUNS" ]
then
    echo "Please specify all parameters (DATASET, WEIGHTED_LOSS, ALPHA, NUM_RUNS)"
    echo "Aborting..."
    exit 1
fi

for i in `seq 1 $NUM_RUNS`;
do
    echo "Starting run $i"
    $DIR/start_k8s_job.sh $i $DATASET $WEIGHTED_LOSS $ALPHA
done
