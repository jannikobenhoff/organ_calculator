#!/bin/bash
#
SBATCH --job-name=<jannik_nnunet> # give your job a name
#SBATCH --nodes=1
SBATCH --cpus-per-task=15
SBATCH --time=03:00:00 # set this time according to your need
SBATCH --mem=200G # how much RAM will your notebook consume? 
SBATCH --gres=gpu:1 # if you need to use a GPU
SBATCH -p sablab-gpu # specify partition
SBATCH --nodelist ai-gpu01
module purge
module load anaconda3
source ../venv/bin/activate

export nnUNet_raw="/home/jao4016/organ_calculator/data/nnUNet_raw"
export nnUNet_preprocessed="/home/jao4016/organ_calculator/data/nnUNet_preprocessed"
export nnUNet_results="/home/jao4016/organ_calculator/data/nnUNet_results"

nnUNetv2_train 101 2d 1