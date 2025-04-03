#!/bin/bash
#
#SBATCH --partition=sablab-gpu   # cluster-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=jannik_segmentation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --time=02:00:00   # HH/MM/SS
#SBATCH --mem=200G   # memory requested, units available: K,M,G,T
#SBATCH --output /home/jao4016/log/job-%j.out
#SBATCH --error /home/jao4016/log/job-%j.err

module purge
module load anaconda3
source ../venv/bin/activate

export nnUNet_raw="/home/jao4016/organ_calculator/data/nnUNet_raw"
export nnUNet_preprocessed="/home/jao4016/organ_calculator/data/nnUNet_preprocessed"
export nnUNet_results="/home/jao4016/organ_calculator/data/nnUNet_results"

nnUNetv2_predict -i ../data/MRI/input/case7 -o ../data/MRI/output/case7 -d 101 -c 2d

nnUNetv2_predict -i ../data/MRI/input/case8 -o ../data/MRI/output/case8 -d 101 -c 2d

nnUNetv2_predict -i ../data/MRI/input/case9 -o ../data/MRI/output/case9 -d 101 -c 2d

nnUNetv2_predict -i ../data/MRI/input/case10 -o ../data/MRI/output/case10 -d 101 -c 2d

# FileNotFoundError: [Errno 2] No such file or directory: '/home/jao4016/organ_calculator/data/nnUNet_results/Dataset101_Totalsegmentator/nnUNetTrainer__nnUNetPlans__2d/fold_2/checkpoint_final.pth'