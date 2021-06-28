#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DATASET=$1
WEIGHTED_LOSS="True"
ALPHAS=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0")
NUM_RUNS="20"

if [ -z "$DATASET" ] && [ -z "$WEIGHTED_LOSS" ] && [ -z "$ALPHA" ] && [ -z "$NUM_RUNS" ]
then
    echo "Please specify all parameters (DATASET, WEIGHTED_LOSS, ALPHA, NUM_RUNS)"
    echo "Aborting..."
    exit 1
fi

for a in "${ALPHAS[@]}";
do
    for i in `seq 1 $NUM_RUNS`;
    do
        echo "Starting run $i"
        $DIR/start_k8s_job.sh $i $DATASET $WEIGHTED_LOSS $a
    done
done
