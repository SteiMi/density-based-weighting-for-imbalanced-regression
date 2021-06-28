#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

WEIGHTED_LOSS=$1
ALPHA=$2
NUM_RUNS=$3

if [ -z "$WEIGHTED_LOSS" ] && [ -z "$ALPHA" ] && [ -z "$NUM_RUNS" ]
then
    echo "Please specify all parameters (WEIGHTED_LOSS, ALPHA, NUM_RUNS)"
    echo "Aborting..."
    exit 1
fi

for i in `seq 1 $NUM_RUNS`;
do
    echo "Starting run $i"
    $DIR/start_k8s_job.sh $i $WEIGHTED_LOSS $ALPHA False
done
