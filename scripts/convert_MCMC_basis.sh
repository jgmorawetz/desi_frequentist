#!/bin/bash
#SBATCH --job-name=chain_convert
#SBATCH --qos=shared
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --array=0-11

PARAMS=(
    "FS LCDM"
    "FS w0waCDM"
    "FS+BAO LCDM"
    "FS+BAO w0waCDM"
    "FS+BAO+CMB LCDM"
    "FS+BAO+CMB w0waCDM"
    "FS+BAO+CMB+DESY5SN LCDM"
    "FS+BAO+CMB+DESY5SN w0waCDM"
    "FS+BAO+CMB+PantheonPlusSN LCDM"
    "FS+BAO+CMB+PantheonPlusSN w0waCDM"
    "FS+BAO+CMB+Union3SN LCDM"
    "FS+BAO+CMB+Union3SN w0waCDM"
)

PARAM_INDEX=$((SLURM_ARRAY_TASK_ID))
PARAM_SET=(${PARAMS[$PARAM_INDEX]})
dataset="${PARAM_SET[0]}"
variation="${PARAM_SET[1]}"

python post_processing_MCMCchains.py --dataset="$dataset" --variation="$variation"