#!/usr/bin/env bash

module purge
module load proxy
module load modtree/gpu
module load cuda/12.9.0
module use /depot/bouman/apps/modules
module load cudnn/9.11.0
module load conda

conda activate mbirjax

for ((views=512; views<65536; views+=512)); do
  echo ""
  echo "Filtering sinogram with shape ($views, 512, 512)"
  python fdk_filter.py "$views" 512 512 || break
done

for ((rows=512; rows<65536; rows+=512)); do
  echo ""
  echo "Filtering sinogram with shape (512, $rows, 512)"
  python fdk_filter.py 512 "$rows" 512 || break
done

for ((channels=512; channels<65536; channels+=512)); do
  echo ""
  echo "Filtering sinogram with shape (512, 512, $channels)"
  python fdk_filter.py 512 512 "$channels" || break
done