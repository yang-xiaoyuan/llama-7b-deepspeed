#!/bin/bash

#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks=4                  # 任务数
#SBATCH --partition=ai             # 分区名称（根据集群修改）

module load CUDA/12.2

deepspeed  --num_nodes=1 --num_gpus=4 deepspeed_train.py 


