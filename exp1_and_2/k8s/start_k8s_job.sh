#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

RUN_NUMBER=$1
DATASET=$2
WEIGHTED_LOSS=$3
ALPHA=$4
NAMESPACE="YOUR_USERNAME"
NAME="denseloss-"

### Check if this script has acceptable parameters

if [ -z "$RUN_NUMBER" ] && [ -z "$DATASET" ] && [ -z "$WEIGHTED_LOSS" ] && [ -z "$ALPHA" ]
then
    echo "Please specify all parameters (RUN_NUMBER, DATASET, WEIGHTED_LOSS, ALPHA)"
    echo "Aborting..."
    exit 1
fi

### Build some variables based on the parameters

WEIGHTED_LOSS_STR=""
ALPHA_STR=$(echo $ALPHA | sed -e "s/\./-/g")
DATASET_STR=$(echo $DATASET | sed -e "s/\_//g" | tr '[:upper:]' '[:lower:]')
NAME="${NAME}${DATASET_STR}-"

if [ "$WEIGHTED_LOSS" = "True" ]
then
    WEIGHTED_LOSS_STR=", \"--weighted_loss\""
    NAME="${NAME}wl-$ALPHA_STR-$RUN_NUMBER"
else
    NAME="$NAME$RUN_NUMBER"
fi

JOB_NAME="$NAME"
echo "Job: $JOB_NAME"

### Delete old job, if there is already one

job_exists=$(kubectl get job -n $NAMESPACE | grep $JOB_NAME)

if [ -n "$job_exists" ]
then
    echo "job already exists"
    echo "Delete old job? [y/n]"
    read delete
    if [ $delete = "y" ]
    then
        kubectl delete job -n $NAMESPACE $JOB_NAME
        echo "Deleted job $JOB_NAME"
    else
        echo "Aborting..."
        exit 1
    fi
fi

start=$(cat $DIR/k8s_job_template.yml | sed -e "s/\${JOB_NAME}/$JOB_NAME/g" -e "s/\${DATASET}/$DATASET/g" -e "s/\${WEIGHTED_LOSS_STR}/$WEIGHTED_LOSS_STR/g" -e "s/\${ALPHA}/$ALPHA/g" -e "s/\${NAMESPACE}/$NAMESPACE/g" | kubectl apply -f -)
echo $start
echo "Started job $JOB_NAME"
