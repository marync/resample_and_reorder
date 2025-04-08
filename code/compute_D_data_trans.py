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
protdir   = str (sys.argv[1])
protein   = str (sys.argv[2])
ligand    = str (sys.argv[3])
rootout   = str (sys.argv[4])
seed      = int (sys.argv[5])

# in directory
rootin = '.'
nAA    = 21

# Seed random number generators
#-------------------------------------------------------------------------------
# only need one RNG for sampling reads
rng = np.random.default_rng (seed)

# Input and output
#-------------------------------------------------------------------------------
# out directory
outdir = os.path.join (rootout, str (seed), protein)
indir  = outdir

# Read in data
#-------------------------------------------------------------------------------

# singles and doubles
y_singles = np.loadtxt (os.path.join (indir, protein + '_Y_singles.txt'))
y_doubles = pickle.load ( open (os.path.join (indir, protein + '_Y_doubles.pkl'), 'rb') )

# singles for ligand
y_ligand = np.loadtxt (os.path.join (indir, ligand + '_Y_singles.txt'))

# dimensions
L_lig, L, n_reps = y_doubles.shape


# Compute ranks for singles and doubles
#-------------------------------------------------------------------------------
# max rank
Mvals = np.sum (~np.isnan (y_singles), axis=0) - 1

print ('Setting M: ' + str (Mvals))

Rmat = np.zeros_like (y_doubles) * np.nan
Dmat = np.zeros_like (y_doubles) * np.nan
for i in range (n_reps) :
    Rmat[:,:,i] = compute_transformed_Rmat (y_doubles[:,:,i], M=Mvals[i])   # R matrix
    Dmat[:,:,i] = compute_D_from_ranks (y_singles[:,i], Rmat[:,:,i],
                                        protein='trans', M=Mvals[i], nAA=nAA) # D matrix

    out_i = os.path.join (outdir, 'rep_' + str (i) )
    if not os.path.isdir (out_i) :
        os.makedirs (out_i)

    np.savetxt ( os.path.join (out_i, protein + '_Rmat.txt'), Rmat[:,:,i] )
    np.savetxt ( os.path.join (out_i, protein + '_Dmat.txt'), Dmat[:,:,i] )


# keep pickled version for ease right now
pickle.dump (Rmat, open (os.path.join (outdir, protein + '_Rmat.pkl'), 'wb'))
pickle.dump (Dmat, open (os.path.join (outdir, protein + '_Dmat_hat.pkl'), 'wb'))


# Now do for ranking of ligands in each pdz background
#-------------------------------------------------------------------------------

Lmat  = np.zeros ( (L, L_lig, n_reps) ) * np.nan
LDmat = np.zeros_like (Lmat)  * np.nan
for i in range (n_reps) :
    yd_i = np.transpose (cp.deepcopy ( y_doubles[:,:,i] ))
    M_i  = np.sum (~np.isnan (y_ligand[:,i])) - 1

    print ('Setting M: ' + str (M_i))
    Lmat[:,:,i]  = compute_transformed_Rmat (yd_i, M=M_i)
    LDmat[:,:,i] = compute_D_from_ranks (y_ligand[:,i], Lmat[:,:,i],
                                         protein='trans', M=M_i, nAA=nAA) # D matrix

    out_i = os.path.join (outdir, 'rep_' + str (i) )
    if not os.path.isdir (out_i) :
        os.makedirs (out_i)

    np.savetxt ( os.path.join (out_i, protein + '_Lmat.txt'), Lmat[:,:,i] )
    np.savetxt ( os.path.join (out_i, protein + '_LDmat.txt'), LDmat[:,:,i] )


# keep pickled version for ease right now
pickle.dump (Lmat, open (os.path.join (outdir, protein + '_Lmat.pkl'), 'wb'))
pickle.dump (LDmat, open (os.path.join (outdir, protein + '_LDmat_hat.pkl'), 'wb'))


