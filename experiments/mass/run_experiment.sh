#!/bin/sh

python train_augnet.py --dir ./saved-outputs/neurips_rebuttal/3l_spatial_time-freq_time-freq_b16_lr0.001_r0.5_c20/ --seed 29 --n_jobs 5 --grouped_subset --batch_size 16 --tf_types spatial --tf_types time-freq --tf_types time-freq --aug_reg 10 --n_jobs 5 --ncopies 20 -d $1