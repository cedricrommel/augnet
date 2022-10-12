#!/bin/bash

# >>> Learning correct invariance experiment <<<
echo "Starting experiment 1: Learning the correct invariance"
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --n_aug_layers 1 --noise 0.5 --backbone cnn --num_channels 2 --num_layers 2 -d $1 --reg 0.2

# >>> Additional experiment: too many layers <<<
echo "Starting additional experiment 1bis: Having too many layers (2 and 4)"
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --n_aug_layers 2 --noise 0.5 --backbone cnn --num_channels 2 --num_layers 2 -d $1 --reg 0.2
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --n_aug_layers 4 --noise 0.5 --backbone cnn --num_channels 2 --num_layers 2 -d $1 --reg 0.2

# >>> Invariance vs Model capacity experiment <<<
echo "Starting experiment 2: Invariance vs Model capacity"

# Baseline
echo "Trainin baseline w/o  augmentation"
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method none --noise 0.5 --backbone mlp --num_channels 1 --num_layers 1 -d $1
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method none --noise 0.5 --backbone mlp --num_channels 2 --num_layers 1 -d $1
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method none --noise 0.5 --backbone mlp --num_channels 2 --num_layers 2 -d $1
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method none --noise 0.5 --backbone mlp --num_channels 2 --num_layers 3 -d $1

# Perfect Augmentation (MLP)
echo "Trainin baseline w/ perfect augmentation"
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method none --noise 0.5 --backbone mlp --num_channels 1 --num_layers 1 --augment -d $1
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method none --noise 0.5 --backbone mlp --num_channels 2 --num_layers 1 --augment -d $1
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method none --noise 0.5 --backbone mlp --num_channels 2 --num_layers 2 --augment -d $1
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method none --noise 0.5 --backbone mlp --num_channels 2 --num_layers 3 --augment -d $1

# AugNet 1 TF (MLP)
echo "Training AugNet with C=10"
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 1 --num_layers 1 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 10
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 2 --num_layers 1 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 10
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 2 --num_layers 2 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 10
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 2 --num_layers 3 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 10

# AugNet 1 TF (MLP)
echo "Training AugNet with C=1"
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 1 --num_layers 1 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 1
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 2 --num_layers 1 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 1
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 2 --num_layers 2 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 1
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 2 --num_layers 3 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 1

echo "Training AugNet with C=4"
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 1 --num_layers 1 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 4
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 2 --num_layers 1 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 4
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 2 --num_layers 2 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 4
python train_augnet.py --dir ./saved-outputs/results/ --lr 0.01 --seed 29 --method augnet --l_types freq --noise 0.5 --backbone mlp --num_channels 2 --num_layers 3 -d $1 --init-mag 0.05 --reg 0.8 --ncopies 4

echo "Numerical results are stored in ./saved-outputs/results/"

# >>> Plotting all results <<<
echo "Plotting results and saving in ./figures/"
python plot_results.py
