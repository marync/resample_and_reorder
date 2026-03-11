#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

NCORES=8
READTHRES=11
ROOTOUT='dimsum'
PROTEIN='fosjun'
SEED=325
ROOTDIR=results/fosjun/${SEED}/fos
OUTDIR=results/fosjun/${SEED}

module load python/anaconda-2023.09

python code/parse_fosjun_dimsum.py $ROOTDIR $OUTDIR

# load conda directory with dimsum installed
#source activate path_to_conda_dir/anaconda/dimsum

cut -f1,2,3 ${OUTDIR}/fosjun_dimsum_nt.txt > ${OUTDIR}/fosjun_rep_0.txt
cut -f1,4,5 ${OUTDIR}/fosjun_dimsum_nt.txt > ${OUTDIR}/fosjun_rep_1.txt
cut -f1,6,7 ${OUTDIR}/fosjun_dimsum_nt.txt > ${OUTDIR}/fosjun_rep_2.txt

for REP in {0..2}
do

	echo $REP
	SEED=$((SEED+2))
	echo $SEED

	DiMSum --experimentDesignPath  ${OUTDIR}/experimentDesign_fosjun_${REP}.txt \
		--wildtypeSequence ACAGACACACTTCAAGCAGAGACAGACCAACTTGAGGACGAGAAGAGCGCACTTCAAACAGAGATCGCAAACCTTCTTAAGGAGAAGGAGAAGCTTATCGCAAGGCTTGAGGAGAAGGTAAAGACACTTAAGGCACAAAACAGCGAGCTTGCAAGCACAGCAAACATGCTTAGGGAGCAAGTAGCACAACTT \
		--projectName fosjun_${SEED}_${REP} --outputPath $OUTDIR --startStage 4 \
		--numCores $NCORES --fitnessMinInputCountAny 11 --fitnessMinOutputCountAny 0 \
		--fitnessNormalise F --countPath ${OUTDIR}/fosjun_rep_${REP}.txt
done
	
SEED=$((SEED+2))
# Run on all replicates together
DiMSum --experimentDesignPath  ${OUTDIR}/experimentDesign_fosjun.txt \
	--wildtypeSequence ACAGACACACTTCAAGCAGAGACAGACCAACTTGAGGACGAGAAGAGCGCACTTCAAACAGAGATCGCAAACCTTCTTAAGGAGAAGGAGAAGCTTATCGCAAGGCTTGAGGAGAAGGTAAAGACACTTAAGGCACAAAACAGCGAGCTTGCAAGCACAGCAAACATGCTTAGGGAGCAAGTAGCACAACTT \
	--projectName fosjun_${SEED} --outputPath $OUTDIR --startStage 4 \
	--numCores $NCORES --fitnessMinInputCountAny 11 --fitnessMinOutputCountAny 0 \
	--fitnessNormalise T --countPath ${OUTDIR}/fosjun_dimsum_nt.txt
