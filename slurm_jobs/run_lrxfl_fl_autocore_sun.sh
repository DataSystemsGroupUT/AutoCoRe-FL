#!/bin/bash
#SBATCH --job-name=run_lrxfl_fl_autocore_sun
#SBATCH --output=logs/run_lrxfl_fl_autocore_sun_%A_%a.log
#SBATCH --error=logs/run_lrxfl_fl_autocore_sun_%A_%a.log
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source /gpfs/helios/home/soliman/logic_sam2/bin/activate

mkdir -p logs
export PYTHONPATH=/gpfs/helios/home/soliman/logic_explained_networks/experiments
python -u -m AutoCore_FL.scripts.run_lrxfl_fl_autocore_sun