#!/bin/bash
#
#SBATCH --job-name=inversion
#SBATCH --output=bin/output_07_inversion_v8_big.out
#SBATCH -t 3-00:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=1
##SBATCH --mem=191000M
##SBATCH -p short
##SBATCH --exclusive

##SBATCH -w node06
#SBATCH --exclude=node[01]
##SBATCH -x node03 
pwd

python -u main_forward_model_v8.py -n_e 55 -n_i 8 -n_p 11 &

wait

