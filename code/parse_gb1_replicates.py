import os
import sys
import numpy as np
import pandas
import pickle

from format_helper_functions import *

# read threshold for N0
readthres = int (sys.argv[1])
rootout   = sys.argv[2]
protein   = sys.argv[3]
seed      = int (sys.argv[4])

# files
indir  = 'data'

outdir = rootout
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

outdir = os.path.join (outdir, str (seed))
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

doubles_f = os.path.join (indir, protein, 'GB1_DMuts_raw.tsv')
singles_f = os.path.join (indir, protein, 'GB1_SMuts_raw.tsv')

# replicates
n_sel = 3  # selection reps
n_neu = 1  # neutral reps
nAA   = 20 # number of amino acids
eta   = 1  # pseudo-count

# columns for selelction
sel_cols = ['Sel' + str (i) for i in range (1,4)]

# AA to numeric dictionary
aadict = make_aa_dict ()

# read in data
doubles_df = pandas.read_csv (doubles_f, delimiter='\t')
singles_df = pandas.read_csv (singles_f, delimiter='\t')

positions = np.sort (np.unique (singles_df['Pos']))
npos      = len (positions)
offset    = np.min (positions)

# total number of mutations
L = (npos*nAA)

# make arrays
S_neu = np.ones ( L, dtype=int ) * -1
S_sel = np.ones ( L, dtype=int) * -1

for index, row in singles_df.iterrows () :
    pos_i = row['Pos'] - offset

    aa_i  = aadict[row['Mut'][-1]]
    idx_i = (pos_i*nAA) + aa_i

    S_neu[idx_i] = row['DNA']
    S_sel[idx_i] = int (np.sum (row[sel_cols]))


# faster recall
D_sel_mat = doubles_df[sel_cols].to_numpy ()

# make arrays
D_neu = np.ones ( (L, L), dtype=int ) * -1
D_sel = np.ones ( (L, L), dtype=int ) * -1
for index, row in doubles_df.iterrows () :
    # extract each mutation
    muts_i1, muts_i2 = row['Mut'].split ('-')

    # positions
    pos_i1 = int (muts_i1[1:-1]) - offset
    pos_i2 = int (muts_i2[1:-1]) - offset

    # amino acids
    aa_i1 = aadict[muts_i1[-1]]
    aa_i2 = aadict[muts_i2[-1]]

    idx_1 = (pos_i1*nAA) + aa_i1
    idx_2 = (pos_i2*nAA) + aa_i2

    D_neu[idx_1, idx_2] = D_neu[idx_2, idx_1] = row['DNA']

    # sum counts
    D_sel[idx_1, idx_2] = D_sel[idx_2, idx_1] = np.sum (D_sel_mat[index,:])


# filter reads based on read count threshold
D_neu[D_neu < readthres] = -1
D_sel[D_neu == -1]       = -1
S_neu[S_neu < readthres] = -1
S_sel[S_neu == -1]       = -1

# save singles
np.savetxt (os.path.join (outdir, 'GB1_S_neutral.txt'), S_neu, fmt='%i')
np.savetxt (os.path.join (outdir, 'GB1_S_selection.txt'), S_sel, fmt='%i')

# save the double info
np.savetxt (os.path.join (outdir, 'GB1_D_neutral.txt'), D_neu, fmt='%i')
np.savetxt (os.path.join (outdir, 'GB1_D_selection.txt'), D_sel, fmt='%i')

# add pseudo-counts
D_neu[D_neu != -1] += eta
D_sel[D_sel != -1] += eta
S_neu[S_neu != -1] += eta
S_sel[S_sel != -1] += eta

# singles
S_neu_f = np.array (S_neu, dtype=float)
S_sel_f = np.array (S_sel, dtype=float)
S_neu_f[S_neu == -1] = np.nan
S_sel_f[S_neu == -1] = np.nan

# doubles
D_neu_f = np.array (D_neu, dtype=float)
D_sel_f = np.array (D_sel, dtype=float)
D_neu_f[D_neu == -1] = np.nan
D_sel_f[D_neu == -1] = np.nan

# save
np.savetxt (os.path.join (outdir, 'GB1_Y_singles.txt'), S_sel_f / S_neu_f)
np.savetxt (os.path.join (outdir, 'GB1_Y_doubles.txt'), D_sel_f / D_neu_f)




