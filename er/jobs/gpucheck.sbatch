#!/bin/bash
#SBATCH -J er_gpuchk
#SBATCH -p mtech
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
nvidia-smi; /scratch/m25csa019/envs/er/bin/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"
