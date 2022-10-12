#!/bin/sh

# No aug
echo "Training trunk model baseline"
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform none --epochs 300 --cosann 300 -d $1 --seed 1
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform none --epochs 300 --cosann 300 -d $1 --seed 29
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform none --epochs 300 --cosann 300 -d $1 --seed 42
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform none --epochs 300 --cosann 300 -d $1 --seed 666
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform none --epochs 300 --cosann 300 -d $1 --seed 777

# Fixed aug
echo "Training baseline w/ fixed data augmentation"
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform fixed --epochs 300 --cosann 300 -d $1 --seed 1 --save-all
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform fixed --epochs 300 --cosann 300 -d $1 --seed 29 --save-all
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform fixed --epochs 300 --cosann 300 -d $1 --seed 42
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform fixed --epochs 300 --cosann 300 -d $1 --seed 666
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform fixed --epochs 300 --cosann 300 -d $1 --seed 777

# RandAugment
echo "Training baseline w/ RandAugment"
python train_augerino_new.py --dir ./saved-outputs/results/--lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform ra --epochs 300 --cosann 300 -d $1 --seed 1
python train_augerino_new.py --dir ./saved-outputs/results/--lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform ra --epochs 300 --cosann 300 -d $1 --seed 29
python train_augerino_new.py --dir ./saved-outputs/results/--lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform ra --epochs 300 --cosann 300 -d $1 --seed 42
python train_augerino_new.py --dir ./saved-outputs/results/--lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform ra --epochs 300 --cosann 300 -d $1 --seed 666
python train_augerino_new.py --dir ./saved-outputs/results/--lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform ra --epochs 300 --cosann 300 -d $1 --seed 777

# AutoAugment
echo "Training baseline w/ AutoAugment"
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform aa --epochs 300 --cosann 300 -d $1 --seed 1
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform aa --epochs 300 --cosann 300 -d $1 --seed 29
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform aa --epochs 300 --cosann 300 -d $1 --seed 42
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform aa --epochs 300 --cosann 300 -d $1 --seed 666
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform aa --epochs 300 --cosann 300 -d $1 --seed 777

# Augerino
echo "Training baseline w/ Augerino"
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --aug-wd 0. --aug_reg 0.05 --method augerino --backbone resnet18 --transform none --aug_lr 0.01 --aug-wd 0. --epochs 300 --cosann 300 --ncopies 20 -d $1 --seed 1
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --aug-wd 0. --aug_reg 0.05 --method augerino --backbone resnet18 --transform none --aug_lr 0.01 --aug-wd 0. --epochs 300 --cosann 300 --ncopies 20 -d $1 --seed 29
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --aug-wd 0. --aug_reg 0.05 --method augerino --backbone resnet18 --transform none --aug_lr 0.01 --aug-wd 0. --epochs 300 --cosann 300 --ncopies 20 -d $1 --seed 42
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --aug-wd 0. --aug_reg 0.05 --method augerino --backbone resnet18 --transform none --aug_lr 0.01 --aug-wd 0. --epochs 300 --cosann 300 --ncopies 20 -d $1 --seed 666
python train_augerino_new.py --dir ./saved-outputs/results/ --lr 1e-3 --wd 2e-2 --aug-wd 0. --aug_reg 0.05 --method augerino --backbone resnet18 --transform none --aug_lr 0.01 --aug-wd 0. --epochs 300 --cosann 300 --ncopies 20 -d $1 --seed 777

# AugNet
echo "Training AugNet"
python train_augerino_new.py --dir ./saved-outputs/results/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 1 -d $1 --save-all
python train_augerino_new.py --dir ./saved-outputs/results/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 29 -d $1 --save-all
python train_augerino_new.py --dir ./saved-outputs/results/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 42 -d $1 # --save-all
python train_augerino_new.py --dir ./saved-outputs/results/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 666 -d $1 # --save-all
python train_augerino_new.py --dir ./saved-outputs/results/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 777 -d $1 # --save-all

# AugNet - lambda sensitivity study
echo "AugNet lambda sensitivity study..."
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 1 -d $1 --aug_reg 0.05
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 29 -d $1 --aug_reg 0.05
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 42 -d $1 --aug_reg 0.05
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 666 -d $1 --aug_reg 0.05
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 777 -d $1 --aug_reg 0.05
#
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 1 -d $1 --aug_reg 0.1
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 29 -d $1 --aug_reg 0.1
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 42 -d $1 --aug_reg 0.1
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 666 -d $1 --aug_reg 0.1
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 777 -d $1 --aug_reg 0.1
#
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 1 -d $1 --aug_reg 1.0
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 29 -d $1 --aug_reg 1.0
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 42 -d $1 --aug_reg 1.0
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 666 -d $1 --aug_reg 1.0
python train_augerino_new.py --dir ./saved-outputs/reg_sensitivity_study/ -d cuda --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --pen-schedule --epochs 300 --cosann 300 --ncopies 20 --seed 777 -d $1 --aug_reg 1.0

# AugNet - C sensivity study wrt inference time
echo "AugNet C sensitivity study..."
python train_augerino_new.py --dir ./saved-outputs/timer_experiment/ --lr 1e-3 --wd 2e-2 --method none --backbone resnet18 --transform fixed --epochs 1 -d $1 --seed 29 --timeit --checkpoint ./saved-outputs/results/baseline_resnet18_fixed_trans_29_trained.pt
python train_augerino_new.py --dir ./saved-outputs/timer_experiment/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --epochs 1 --seed 29 -d $1 --ncopies 1 --timeit --checkpoint ./saved-outputs/results/augnet-c20-non-aff-aff-aff_resnet18_no_trans_29_trained.pt
python train_augerino_new.py --dir ./saved-outputs/timer_experiment/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --epochs 1 --seed 29 -d $1 --ncopies 4 --timeit --checkpoint ./saved-outputs/results/augnet-c20-non-aff-aff-aff_resnet18_no_trans_29_trained.pt
python train_augerino_new.py --dir ./saved-outputs/timer_experiment/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --epochs 1 --seed 29 -d $1 --ncopies 10 --timeit --checkpoint ./saved-outputs/results/augnet-c20-non-aff-aff-aff_resnet18_no_trans_29_trained.pt
python train_augerino_new.py --dir ./saved-outputs/timer_experiment/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --epochs 1 --seed 29 -d $1 --ncopies 20 --timeit --checkpoint ./saved-outputs/results/augnet-c20-non-aff-aff-aff_resnet18_no_trans_29_trained.pt
python train_augerino_new.py --dir ./saved-outputs/timer_experiment/ --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --epochs 1 --seed 29 -d $1 --ncopies 40 --timeit --checkpoint ./saved-outputs/results/augnet-c20-non-aff-aff-aff_resnet18_no_trans_29_trained.pt

# AugNet - C vs performance study
python eval_perf_vs_C.py --method augnet --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --seed 29 -d $1 --pen-schedule

# AugNet - model invariance study
echo "AugNet model invariance study..."
python eval_model_invariance.py --backbone resnet18 --n_layers 3 --l_types non-aff --l_types aff --l_types aff --lr 0.001 --wd 2e-2 --aug-wd 0. --aug_reg 0.5 --seed 29 -d $1

echo "Plotting results"
python plot_results.py
