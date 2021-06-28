#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

WEIGHTED_LOSS="True"
ALPHAS=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0" "1.2" "1.4" "1.6" "1.8" "2.0" "2.2" "2.4" "2.6" "2.8" "3.0" "3.2" "3.4" "3.6" "3.8" "4.0")
NUM_RUNS="20"

if [ -z "$WEIGHTED_LOSS" ] && [ -z "$ALPHA" ] && [ -z "$NUM_RUNS" ]
then
    echo "Please specify all parameters (WEIGHTED_LOSS, ALPHA, NUM_RUNS)"
    echo "Aborting..."
    exit 1
fi

for a in "${ALPHAS[@]}";
do
    for i in `seq 1 $NUM_RUNS`;
    do
        echo "Starting run $i"
        $DIR/start_k8s_job.sh $i $WEIGHTED_LOSS $a False
    done
done
