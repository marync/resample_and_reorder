import os
import sys
import copy as cp
import numpy as np
import pickle

# my functions
sys.path.insert(1, '../code/')

from rank_functions import compute_transformed_Rmat
from D_functions import compute_D_from_ranks

# Hardcoded parameters
#-------------------------------------------------------------------------------
protein   = str (sys.argv[1])
nsims     = int (sys.argv[2])
rootout   = str (sys.argv[3])
seed      = int (sys.argv[4])

# how missing data is coded
if protein in ['fosjun', 'GB1'] :
    NA = -1
else :
    NA = None

if protein == 'fosjun' :
    dprot = protein
    nAA   = 21
else :
    dprot = None
    nAA = 20

# Seed random number generators
#-------------------------------------------------------------------------------
# only need one RNG for sampling reads
rng = np.random.default_rng (seed)

# Input and output
#-------------------------------------------------------------------------------
# out directory
outdir = rootout
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

outdir = os.path.join (outdir, str (seed))
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

indir = outdir
# Read in data
#-------------------------------------------------------------------------------
s_neu = np.loadtxt (os.path.join (indir, protein + '_S_neutral.txt'))
s_sel = np.loadtxt (os.path.join (indir, protein + '_S_selection.txt'))

if protein in ['fosjun','fos','jun'] :
    d_neu     = pickle.load ( open (os.path.join (indir, protein + '_D_neutral.pkl'), 'rb') )
    y_doubles = pickle.load ( open (os.path.join (indir, protein + '_Y_doubles.pkl'), 'rb') )
elif protein == 'GB1' :
    d_neu     = np.loadtxt (os.path.join (indir, protein + '_D_neutral.txt'))
    y_doubles = np.loadtxt (os.path.join (indir, protein + '_Y_doubles.txt'))

# fitness estimates
y_singles = np.loadtxt (os.path.join (indir, protein + '_Y_singles.txt'))
print (y_singles.ndim)

# position info
if y_singles.ndim == 2 :
    L, n_reps = y_singles.shape
else :
    L      = len (y_singles)
    n_reps = 1

if n_reps == 1 :
    d_sel = np.loadtxt (os.path.join (indir, protein + '_D_selection.txt'))
else :
    d_sel = pickle.load ( open (os.path.join (indir, protein + '_D_selection.pkl'), 'rb') )


# Compute ranks for singles and doubles
#-------------------------------------------------------------------------------
# max rank
if np.logical_and (n_reps > 1, protein == 'GB1') :
    M = np.max (np.sum (~np.isnan (y_singles), axis=0)) - 1 - (nAA - 1)
elif protein == 'GB1' :
    M = np.max (np.sum (~np.isnan (y_doubles), axis=0)) - 1
elif protein == 'fosjun' and n_reps > 1 :
    M = int (np.ceil (np.max (np.sum (~np.isnan (y_singles), axis=0)) / 2)) - 1

print ('Setting M: ' + str (M))

Rmat    = np.zeros_like (y_doubles) * np.nan
Dmat    = np.zeros_like (y_doubles) * np.nan
Dmatbar = np.zeros_like (y_doubles) * np.nan

if n_reps > 1 :
    for i in range (n_reps) :
        Rmat[:,:,i] = compute_transformed_Rmat (y_doubles[:,:,i], M=M)        # R matrix
        Dmat[:,:,i] = compute_D_from_ranks (y_singles[:,i], Rmat[:,:,i],
                                            protein=dprot, M=M, nAA=nAA) # D matrix

        out_i = os.path.join (outdir, 'rep_' + str (i) )
        if not os.path.isdir (out_i) :
            os.makedirs (out_i)

        np.savetxt ( os.path.join (out_i, protein + '_Rmat.txt'), Rmat[:,:,i] )
        np.savetxt ( os.path.join (out_i, protein + '_Dmat.txt'), Dmat[:,:,i] )

else :
    Rmat = compute_transformed_Rmat (y_doubles, M=M)                  # R matrix
    Dmat = compute_D_from_ranks (y_singles, Rmat, protein=dprot, M=M) # D matrix

    np.savetxt ( os.path.join (outdir, protein + '_Rmat.txt'), Rmat )
    np.savetxt ( os.path.join (outdir, protein + '_Dmat.txt'), Dmat )

# keep pickled version for ease right now
pickle.dump (Rmat, open (os.path.join (outdir, protein + '_Rmat.pkl'), 'wb'))
pickle.dump (Dmat, open (os.path.join (outdir, protein + '_Dmat.pkl'), 'wb'))
