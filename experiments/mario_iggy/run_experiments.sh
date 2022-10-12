#!/bin/bash

echo "### Starting Mario-Iggy experiments ###"

echo "-- Step 1/4: Running Augerino training"
python augerino_training.py --prefix augerino --reg 0.2

echo "-- Step 2/4: Running AugNet training"
python augnet_training.py --wd 1. --lr 0.0005 --reg 0.5 --prefix augnet

echo "-- Step 3/4: Running AugNet training without regularization"
python augnet_training.py --wd 1. --lr 0.0005 --reg 0. --prefix augnet_no_reg

echo "-- Step 4/4: Running AugNet training with Augerino regularizer"
python augnet_training.py --wd 1. --lr 0.0005 --reg 0.5 --pen incomplete --prefix augnet_incomp_reg

echo "Plotting results"
python plot_results.py

echo "Plots can be found in ./figures, while obtained results are stored in ./saved-outputs"

