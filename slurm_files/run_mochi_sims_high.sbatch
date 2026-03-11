#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# SIMULATION
SEED=777
OUTDIR=results/simulations/${SEED}
DATADIR=results/simulations/${SEED}/fitness_dimsum

module load python/anaconda-2023.09

# parse dimsum output
python code/parse_dimsum_for_mochi.py $DATADIR $OUTDIR

# load mochi conda environment
#source activate path_to_conda_dir/anaconda/mochi/

PROJECT=mochi_sigmoid
SEED_MOCHI=1119

run_mochi.py --model_design ${OUTDIR}/model_design_sigmoid.txt --output_directory $OUTDIR --holdout_orders 2 \
	     --seed $SEED_MOCHI --project_name $PROJECT
