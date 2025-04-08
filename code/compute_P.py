import os
import sys
import copy as cp
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle

# my functions
sys.path.insert(1, '../code/')

from rank_functions import compute_rank_sample, compute_ranks_unique
from P_functions import compute_p_values, compute_p_values_integrate, compute_p_values_two_sided

# Hardcoded parameters
#-------------------------------------------------------------------------------
rootdir = str (sys.argv[1])
seed    = int (sys.argv[2])
protein = str (sys.argv[3])

if len (sys.argv) == 5 :
    rep = int (sys.argv[-1])
    nAA = 21
else :
    rep = None
    nAA = 20

# rng
rng = np.random.default_rng (seed)

# Input and output
#-------------------------------------------------------------------------------
if rep is not None :
    indir = os.path.join (rootdir, str (seed), 'rep_' + str (rep))
else :
    indir = os.path.join (rootdir, str (seed))

# output to the input directory
outdir = indir

print ('input directory for D')
print (indir)

# observed D matrix
Rmat = np.loadtxt (os.path.join (indir, protein + '_Rmat.txt'))
Dmat = np.loadtxt (os.path.join (indir, protein + '_Dmat.txt'))

L, L = Rmat.shape

# singles
Y_singles_mat = np.loadtxt (os.path.join (indir, protein + '_Y_singles.txt'))
if rep is not None :
    Y_singles = Y_singles_mat[:,rep]
else :
    Y_singles = cp.deepcopy (Y_singles_mat)

# open the simulated read data
filename   = os.path.join (outdir, protein + '_Dmats.pkl')
fileObject = open (filename, 'rb')
Dmats_sim = pickle.load (fileObject)
fileObject.close ()

# open the simulated read data
filename   = os.path.join (outdir, protein + '_Dmats_unsorted.pkl')
fileObject = open (filename, 'rb')
Dmats_sim_unordered = pickle.load (fileObject)
fileObject.close ()

# make sure that any singles which should be removed are
bad = np.where (np.sum (np.isnan (Rmat), axis=0) == L)[0]
print ('bad')
print (bad)
Y_singles[bad] = np.nan



# Compute the p-values 
#-------------------------------------------------------------------------------

# compute the rank of the singles
order_ys = compute_rank_sample (Y_singles, rng)

# p-values
P, Pasym, Psym = compute_p_values (D_obs=Dmat, D_sim=Dmats_sim, rank_singles=order_ys)
# integrated p-value
Pint = compute_p_values_integrate (D_obs=Dmat,
                                   D_sim=Dmats_sim,
                                   D_sim_u=Dmats_sim_unordered,
                                   row_rank=order_ys)

Ptwo = compute_p_values_two_sided (D_obs=Dmat,
                                   D_sim=Dmats_sim,
                                   rank_singles=order_ys)

# save
np.savetxt (os.path.join (indir, protein + '_P.txt'), P)
np.savetxt (os.path.join (indir, protein + '_Pasym.txt'), Pasym)
np.savetxt (os.path.join (indir, protein + '_Psym.txt'), Psym)
np.savetxt (os.path.join (indir, protein + '_Pint.txt'), Pint)
np.savetxt (os.path.join (indir, protein + '_Ptwo.txt'), Ptwo)




