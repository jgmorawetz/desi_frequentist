#!/bin/bash
#SBATCH --job-name=prof_FS++Union3SN
#SBATCH --account=rrg-wperciva
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15G
#SBATCH --time=96:00:00
#SBATCH --array=0-255


module load julia

# Iterates through different param/dataset/variation/index combinations, submits parallel jobs for each
VERSIONS=("ln10As FS+BAO+CMB+Union3SN LCDM" "ln10As FS+BAO+CMB+Union3SN w0waCDM" "H0 FS+BAO+CMB+Union3SN LCDM" "H0 FS+BAO+CMB+Union3SN w0waCDM" "omegac FS+BAO+CMB+Union3SN LCDM" "omegac FS+BAO+CMB+Union3SN w0waCDM" "w0 FS+BAO+CMB+Union3SN w0waCDM" "wa FS+BAO+CMB+Union3SN w0waCDM")
RUN_NUMBERS=({1..32})
N_RUNS=100
N_PROFILE=32

VERSION_INDEX=$((SLURM_ARRAY_TASK_ID / N_PROFILE))
RUN_NUMBER_INDEX=$((SLURM_ARRAY_TASK_ID % N_PROFILE))
LABELS=(${VERSIONS[$VERSION_INDEX]})
PARAM_LABEL="${LABELS[0]}"
DATASET="${LABELS[1]}"
VARIATION="${LABELS[2]}"
PARAM_INDEX=${RUN_NUMBERS[$RUN_NUMBER_INDEX]}

if [ "$PARAM_LABEL" == "ln10As" ]; then
    PARAM_LOWER=2.5
    PARAM_UPPER=3.5
elif [ "$PARAM_LABEL" == "H0" ]; then
    PARAM_LOWER=50
    PARAM_UPPER=80
elif [ "$PARAM_LABEL" == "omegac" ]; then
    PARAM_LOWER=0.09
    PARAM_UPPER=0.2
elif [ "$PARAM_LABEL" == "w0" ]; then
    PARAM_LOWER=-2
    PARAM_UPPER=0.5
elif [ "$PARAM_LABEL" == "wa" ]; then
    PARAM_LOWER=-3
    PARAM_UPPER=1.64
fi

julia profile_likelihoods_1D_DESIY1FullShape.jl --n_runs=$N_RUNS --param_label="$PARAM_LABEL" --param_lower=$PARAM_LOWER --param_upper=$PARAM_UPPER --n_profile=$N_PROFILE --param_index=$PARAM_INDEX --dataset="$DATASET" --variation="$VARIATION"