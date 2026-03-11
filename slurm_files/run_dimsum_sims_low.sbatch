#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

NCORES=8
OUTDIR=results/simulations/888
READTHRES=11
PROTEIN='sim'
SEED=888
WT0=175925
WT1=147160

PROJECT='fitness_dimsum'

module load python/anaconda-2023.09

python code/parse_gb1_dimsum.py ${OUTDIR} ${OUTDIR} $PROTEIN $WT0 $WT1

# load conda directory with dimsum installed
#source activate path_to_conda_dir/anaconda/dimsum

DiMSum --experimentDesignPath ${OUTDIR}/experimentDesign_sims.txt \
	--wildtypeSequence CAATATAAGCTTATCCTTAACGGTAAGACACTTAAGGGTGAGACAACAACAGAGGCAGTAGACGCAGCAACAGCAGAGAAGGTATTTAAGCAATATGCAAACGACAACGGTGTAGACGGTGAGTGGACATATGACGACGCAACAAAGACATTTACAGTAACAGAG \
	--projectName $PROJECT --outputPath $OUTDIR --startStage 4 \
	--numCores $NCORES --fitnessMinInputCountAny 12 --fitnessNormalise F --countPath ${OUTDIR}/${PROTEIN}_dimsum_nt.txt
