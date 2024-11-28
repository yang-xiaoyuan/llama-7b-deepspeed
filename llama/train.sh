#!/bin/bash

#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks=1                  # 任务数
#SBATCH --partition=ai             # 分区名称（根据集群修改）

module load CUDA/12.2

export CUDA_VISIBLE_DEVICES=0 
python train.py
