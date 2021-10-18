#!/bin/bash

MODES=('train')
# MODES=('train' 'test')
gpu=0
# WEAK_LABEL_RATIOS=(0.01 0.001 0.0001)
WEAK_LABEL_RATIOS=(0.1 0.01)

# TODO: ablation study
# num_k_query_pts
# how to concat features

for mode in "${MODES[@]}"; do
	for weak_label_ratio in "${WEAK_LABEL_RATIOS[@]}"; do
		time python -B main_S3DIS_SQN.py \
		--gpu ${gpu} \
		--mode ${mode} \
		--test_area 5 \
		--weak_label_ratio ${weak_label_ratio}
	done
done