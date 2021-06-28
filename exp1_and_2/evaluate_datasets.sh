
DATASETS=("abalone" "concreteStrength" "delta_ailerons" "boston"
          "available_power" "servo" "bank8FM" "machineCpu" "airfoild"
          "a2" "a3" "a1" "cpu_small" "acceleration" "maximal_torque"
          "a4" "a5" "a7" "fuel_consumption_country" "a6")

for ds in "${DATASETS[@]}";
do
    echo "$ds"
    python3 evaluate.py --save-paths "YOUR_BASEPATH/models/weighted-loss/real/${ds}" "YOUR_BASEPATH/models" --plot-prefix "${ds}"
done
