#!/bin/bash
#
#SBATCH --partition=sablab-gpu   # cluster-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=jannik_med_synth_gan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --time=05:00:00   # HH/MM/SS
#SBATCH --mem=200G   # memory requested, units available: K,M,G,T
#SBATCH --output /home/jao4016/log/job-%j.out
#SBATCH --error /home/jao4016/log/job-%j.err

module purge
module load anaconda3
source ../../venv/bin/activate

cd ../inference || exit

python -u generate.py
