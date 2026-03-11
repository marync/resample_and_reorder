#!/bin/bash

#SBATCH --account=pi-amurugan
#SBATCH --partition=caslake
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

SEED=114
OUTDIR=results/GB1/827
DATADIR=results/GB1/827
PROJECT=gb1_dimsum

module load python/anaconda-2023.09

# load conda directory
# source activate path_to_conda_dir/anaconda/mochi/

# parse dimsum output for mochi
python code/parse_dimsum_for_mochi.py ${DATADIR}/${PROJECT} $DATADIR

# THREE STATE MODEL
run_mochi.py --model_design $DATADIR/model_design.txt --output_directory $OUTDIR --holdout_orders 2 \
	     --seed $SEED --project_name $PROJECT


# TWO STATE MODEL
SEED=225
PROJECT=gb1_twostate
run_mochi.py --model_design $DATADIR/model_design_twostate.txt --output_directory $OUTDIR --holdout_orders 2 \
	     --seed $SEED --project_name $PROJECT
