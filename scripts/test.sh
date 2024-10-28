#!/bin/bash
#
SBATCH --job-name=<jannik_test> # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00 # set this time according to your need
#SBATCH --mem=3GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
module purge
module load anaconda3
source ~/myenv/bin/activate
# Or if in your home dir: source ~/myenv/bin/activate
python3 ./a.py
