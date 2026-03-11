#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load python/anaconda-2023.09

NCORES=16
NSIMS=500
INDIR=data
PROTEIN='sim'

# UNIFORM RANGE SIMULATIONS
ROOTDIR=results/simulations
ALLEQUAL='False'
SEED=999 # 666 wt = olson value, uniform range [-3.5,3.5], ns=0
SIGMA=2
READTHRES=11
TWOSTATE='True'
THETA=0
SFACTOR=10
RANGE='False'
NS=.0007

python code/parse_gb1_replicates.py $READTHRES $ROOTDIR $PROTEIN $SEED
python code/simulate_two_state_range.py $ROOTDIR $INDIR $SEED $SFACTOR $SIGMA $READTHRES $TWOSTATE $ALLEQUAL $RANGE $THETA $NS 
python code/simulate_fitness_values.py $PROTEIN $NSIMS $ROOTDIR $SEED $READTHRES
python code/parallel_compute_rank_matrices.py $NCORES $NSIMS $ROOTDIR $SEED $PROTEIN
python code/parallel_compute_D.py $NCORES $ROOTDIR $SEED $PROTEIN
python code/compute_P.py $ROOTDIR $SEED $PROTEIN

