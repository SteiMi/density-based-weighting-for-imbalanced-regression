#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# For SMOGN
# DATASETS=("abalone_smogn" "concreteStrength_smogn" "delta_ailerons_smogn" "boston_smogn"
#           "available_power_smogn" "servo_smogn" "bank8FM_smogn" "machineCpu_smogn"
#           "airfoild_smogn" "a2_smogn" "a3_smogn" "a1_smogn" "cpu_small_smogn"
#           "acceleration_smogn" "maximal_torque_smogn" "a4_smogn" "a5_smogn"
#           "a7_smogn" "fuel_consumption_country_smogn" "a6_smogn")
# WEIGHTED_LOSS="False"
# ALPHAS=("0.0")

# For SMOGN_DW
DATASETS=("abalone_smogn_dw" "concreteStrength_smogn_dw" "delta_ailerons_smogn_dw" "boston_smogn_dw"
          "available_power_smogn_dw" "servo_smogn_dw" "bank8FM_smogn_dw" "machineCpu_smogn_dw"
          "airfoild_smogn_dw" "a2_smogn_dw" "a3_smogn_dw" "a1_smogn_dw" "cpu_small_smogn_dw"
          "acceleration_smogn_dw" "maximal_torque_smogn_dw" "a4_smogn_dw" "a5_smogn_dw"
          "a7_smogn_dw" "fuel_consumption_country_smogn_dw" "a6_smogn_dw")
WEIGHTED_LOSS="False"
ALPHAS=("0.0")

# For DenseLoss and doing nothing
DATASETS=("abalone" "concreteStrength" "delta_ailerons" "boston"
          "available_power" "servo" "bank8FM" "machineCpu" "airfoild"
          "a2" "a3" "a1" "cpu_small" "acceleration" "maximal_torque"
          "a4" "a5" "a7" "fuel_consumption_country" "a6")
WEIGHTED_LOSS="True"
ALPHAS=("0.0" "1.0" )


NUM_RUNS="20"

if [ -z "$DATASETS" ] && [ -z "$WEIGHTED_LOSS" ] && [ -z "$ALPHA" ] && [ -z "$NUM_RUNS" ]
then
    echo "Please specify all parameters (DATASETS, WEIGHTED_LOSS, ALPHA, NUM_RUNS)"
    echo "Aborting..."
    exit 1
fi

for dataset in "${DATASETS[@]}";
do
    for a in "${ALPHAS[@]}";
    do
        for i in `seq 1 $NUM_RUNS`;
        do
            echo "Starting run $i"
            $DIR/start_k8s_job.sh $i $dataset $WEIGHTED_LOSS $a
        done
    done
done
