#!/bin/bash

MODES=('train')
# MODES=('train' 'test')
gpu=0
BATCH_SIZES=(16 10 8 6 4)
num_points=40960
max_epoch=400
# WEAK_LABEL_RATIOS=(0.1 0.01 0.001 0.0001)
# WEAK_LABEL_RATIOS=(0.1 0.01)
WEAK_LABEL_RATIOS=(0.001)

# TODO: ablation study
# num_k_query_pts
# how to concat features

for mode in "${MODES[@]}"; do
	for weak_label_ratio in "${WEAK_LABEL_RATIOS[@]}"; do
			for batch_size in "${BATCH_SIZES[@]}"; do

			echo "batch_size: ${batch_size}"
			echo "num_points: ${num_points}"
			echo "max_epoch: ${max_epoch}"
			echo "weak_label_ratio: ${weak_label_ratio}"

			time python -B main_S3DIS_SQN.py \
			--gpu ${gpu} \
			--mode ${mode} \
			--test_area 5 \
			--batch_size ${batch_size} \
			--num_points ${num_points} \
			--max_epoch ${max_epoch} \
			--weak_label_ratio ${weak_label_ratio}
		done
	done
done

echo "finish training."