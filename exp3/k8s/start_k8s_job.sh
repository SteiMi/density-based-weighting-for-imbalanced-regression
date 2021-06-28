#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

RUN_NUMBER=$1
WEIGHTED_LOSS=$2
ALPHA=$3
TEST_ON_VAL=${4:False}
NAMESPACE="YOUR_NAMESPACE"
MIN_YEAR=1981
MAX_YEAR=2014
MAX_TRAIN_YEAR=2005

### Check if this script has acceptable parameters

if [ -z "$RUN_NUMBER" ] && [ -z "$WEIGHTED_LOSS" ] && [ -z "$ALPHA" ]
then
    echo "Please specify all parameters (RUN_NUMBER, WEIGHTED_LOSS, ALPHA)"
    echo "Aborting..."
    exit 1
fi

### Build some variables based on the parameters

ALPHA_STR=$(echo $ALPHA | sed -e "s/\./-/")

if [ "$TEST_ON_VAL" = "True" ]
then
    NAME="hpsearch-"
    ### NOTE: THIS HERE SPECIFIES THE VALIDATION DATASET!!!
    # We didn't end up needing this.
    MAX_YEAR=2005
    MAX_TRAIN_YEAR=2000
else
    NAME=""
fi

if [ "$WEIGHTED_LOSS" = "True" ]
then
    NAME="${NAME}wl-$ALPHA_STR-$RUN_NUMBER"
else
    NAME="$NAME$RUN_NUMBER"
fi

JOB_NAME="deepsd-$NAME"
echo "Job: deepsd-$NAME"

### Check whether the config for these parameters already exist

CONFIG_DIR="$DIR/../configs/"
CONFIG_NAME="config-mini-$NAME.ini"
CONFIG_PATH="$CONFIG_DIR$CONFIG_NAME"

if [ -f "$CONFIG_PATH" ]
then
    echo "Config file $CONFIG_PATH already exists"
    echo "Aborting..."
    exit 1
fi

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

### Create config

start=$(cat $DIR/config_template.ini | sed -e "s/\${NAME}/$NAME/" -e "s/\${WEIGHTED_LOSS}/$WEIGHTED_LOSS/" -e "s/\${ALPHA}/$ALPHA/" -e "s/\${MIN_YEAR}/$MIN_YEAR/" -e "s/\${MAX_YEAR}/$MAX_YEAR/" -e "s/\${MAX_TRAIN_YEAR}/$MAX_TRAIN_YEAR/" > $CONFIG_PATH)

### Create ConfigMap in Kubernetes
CONFIGMAP_NAME=$(echo "deepsd-$CONFIG_NAME" | sed -e "s/\.ini//")
kubectl create configmap -n $NAMESPACE $CONFIGMAP_NAME --from-file=$CONFIG_PATH

start=$(cat $DIR/k8s_job_template.yml | sed -e "s/\${JOB_NAME}/$JOB_NAME/" -e "s/\${NAMESPACE}/$NAMESPACE/" -e "s/\${CONFIGMAP_NAME}/$CONFIGMAP_NAME/" -e "s/\${CONFIG_NAME}/$CONFIG_NAME/" | kubectl apply -f -)
echo $start
echo "Started job $JOB_NAME"
