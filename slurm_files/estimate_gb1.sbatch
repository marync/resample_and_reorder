#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

module load python/anaconda-2023.09

ROOTDIR=results/GB1 # path to results
NCORES=16
NSIMS=1000
PROTEIN='GB1'
READTHRES=21
SEED=827

python code/parse_gb1_replicates.py $READTHRES $ROOTDIR $PROTEIN $SEED
python code/parse_gb1_dimsum.py ${ROOTDIR}/${SEED} ${ROOTDIR}/${SEED}
python code/simulate_fitness_values.py $PROTEIN $NSIMS $ROOTDIR $SEED $READTHRES 
python code/compute_D_data.py $PROTEIN $NSIMS $ROOTDIR $SEED
python code/parallel_compute_rank_matrices.py $NCORES $NSIMS $ROOTDIR $SEED $PROTEIN
python code/parallel_compute_D.py $NCORES $ROOTDIR $SEED $PROTEIN
python code/compute_P.py $ROOTDIR $SEED $PROTEIN
