#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

module load python/anaconda-2023.09

NCORES=16
NSIMS=1000
ROOTDIR=simulations/threestate
SEED=953 
ALLEQUAL='True'
INDIR=data
PROTEIN='sim'
SIGMA=0
READTHRES=21
TWOSTATE='False'
SFACTOR=1
RANGE='False'

python code/parse_gb1_replicates.py $READTHRES $ROOTDIR $PROTEIN $SEED
python code/simulate_two_state_range.py $ROOTDIR $INDIR $SEED $SFACTOR $SIGMA $READTHRES $TWOSTATE $ALLEQUAL $RANGE
python code/simulate_fitness_values.py $PROTEIN $NSIMS $ROOTDIR $SEED $READTHRES
python code/parallel_compute_rank_matrices.py $NCORES $NSIMS $ROOTDIR $SEED $PROTEIN
python code/parallel_compute_D.py $NCORES $ROOTDIR $SEED $PROTEIN
python code/compute_P.py $ROOTDIR $SEED $PROTEIN

