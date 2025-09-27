#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100_80GB:2  # Request 2 A100 80GB GPUs
#SBATCH --mem=100G
#SBATCH --time=60:00:00
#SBATCH --output=dpo_run.out
#SBATCH --error=dpo_run.err
#SBATCH --partition=array,shire-general  # Use partitions with broad GPU access

# Load bash config and activate conda environment
source /home/arnaik/.bashrc
conda activate handbook

# Move to your project directory
cd /home/arnaik/OracleProject/alignment-handbook

# Run your DPO training script
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_dpo.py \
    recipes/qwen3-4b-instruct/transfer_learning_dpo_full.yaml