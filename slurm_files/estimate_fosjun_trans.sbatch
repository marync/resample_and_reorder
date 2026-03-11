#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=5:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load python/anaconda-2023.09

ROOTDIR=results/fosjun # path to results directory
NCORES=16
NSIMS=1000
PROTDIR='fosjun'
PROTEIN='fos'
LIGAND='jun'
READTHRES=11
SEED=325

python code/parse_fosjun_trans.py $READTHRES $ROOTDIR $PROTEIN $SEED
python code/simulate_fitness_trans.py $PROTDIR $PROTEIN $LIGAND $READTHRES $NSIMS $ROOTDIR $SEED
python code/compute_D_data_trans.py $PROTDIR $PROTEIN $LIGAND $ROOTDIR $SEED

for REP in {0..2};
do
	python code/parallel_compute_rank_matrices_trans.py $PROTEIN $LIGAND $ROOTDIR $SEED $NCORES $REP;
	python code/parallel_compute_D_trans.py $PROTEIN $LIGAND $ROOTDIR $SEED $NCORES $REP;
	python code/compute_P_trans.py $PROTDIR $PROTEIN $LIGAND $ROOTDIR $SEED $REP;
done

