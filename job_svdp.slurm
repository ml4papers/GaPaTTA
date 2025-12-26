#!/bin/bash
#SBATCH --job-name=svdp_single
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/fs72306/leijx0408/SVDP_cross_stage_consistency_warmup_vsc/logs/svdp_%j.out
#SBATCH --error=/home/fs72306/leijx0408/SVDP_cross_stage_consistency_warmup_vsc/logs/svdp_%j.err

# check CUDA
echo "===GPU Info==="
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
echo "=== GPU Stat==="
gpustat
# Go to project dir
export WANDB_API_KEY=c88bced4431c832236b5b9754f4f72b2d765efe2
export PYTHONPATH=.

# Run your training script
bash bash/svdp.sh

