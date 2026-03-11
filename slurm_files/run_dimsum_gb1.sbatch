#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

NCORES=8
OUTDIR=results/GB1/827
READTHRES=21
ROOTOUT='dimsum'
PROTEIN='GB1'
SEED=114

module load python/anaconda-2023.09

# Run if you have not already parsed GB1 raw counts
#python code/parse_gb1_replicates.py $READTHRES $OUTDIR $PROTEIN $SEED 
#python code/parse_gb1_dimsum.py ${OUTDIR} ${OUTDIR}

# Load conda environment with dimsum installed
#source activate path_to_conda_directory/anaconda/dimsum

DiMSum --experimentDesignPath ${OUTDIR}/experimentDesign_GB1.txt \
	--wildtypeSequence CAATATAAGCTTATCCTTAACGGTAAGACACTTAAGGGTGAGACAACAACAGAGGCAGTAGACGCAGCAACAGCAGAGAAGGTATTTAAGCAATATGCAAACGACAACGGTGTAGACGGTGAGTGGACATATGACGACGCAACAAAGACATTTACAGTAACAGAG \
	--projectName gb1_dimsum --outputPath $OUTDIR --startStage 4 \
	--numCores $NCORES --fitnessMinInputCountAny 22 --fitnessNormalise F --countPath ${OUTDIR}/GB1_dimsum_nt.txt
