#!/usr/bin/env bash

module purge
module load proxy
module load modtree/gpu
module load cuda/12.9.0
module use /depot/bouman/apps/modules
module load cudnn/9.11.0
module load conda

conda activate mbirjax

for ((size=256; size<65536; size+=256)); do
  echo "$size"
done