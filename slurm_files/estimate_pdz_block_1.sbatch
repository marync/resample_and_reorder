#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load python/anaconda-2023.09

ROOTDIR=results/fuzzy # path to results

NCORES=16
NSIMS=1000
PROTDIR='fuzzy'
READTHRES=11
MISSING='.05'
POSTHRES=20

# BLOCK 1
PROTEIN='block_1'
LIGAND='cript_1'
SEED=1255

python code/parse_pdz_fuzzy_block.py $READTHRES $POSTHRES $MISSING $ROOTDIR $SEED $PROTEIN 1
python code/simulate_fitness_trans.py $PROTDIR $PROTEIN $LIGAND $READTHRES $NSIMS $ROOTDIR $SEED 
python code/compute_D_data_trans.py $PROTDIR $PROTEIN $LIGAND $ROOTDIR $SEED

for REP in {0..2};
do
	python code/parallel_compute_rank_matrices_trans.py $PROTEIN $LIGAND $ROOTDIR $SEED $NCORES $REP;
	python code/parallel_compute_D_trans.py $PROTEIN $LIGAND $ROOTDIR $SEED $NCORES $REP;
	python code/compute_P_trans.py $PROTDIR $PROTEIN $LIGAND $ROOTDIR $SEED $REP;
done
