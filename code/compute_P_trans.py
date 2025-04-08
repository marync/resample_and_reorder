import os
import sys
import copy as cp
import numpy as np
import pickle

# my functions
sys.path.insert(1, '../code/')

from rank_functions import compute_mean_ranks 
from P_functions import compute_p_values_trans, compute_p_values_trans_symmetric, compute_p_values_trans_two_sided, compute_p_values_trans_integrate

# Hardcoded parameters
#-------------------------------------------------------------------------------
protdir = str (sys.argv[1])
protein = str (sys.argv[2])
ligand  = str (sys.argv[3])
rootdir = str (sys.argv[4])
seed    = int (sys.argv[5])

if len (sys.argv) == 7 :
    rep = int (sys.argv[6])
else :
    rep = None

#combine = True
combine   = False
integrate = True

# rng
rng = np.random.default_rng (seed)

# Input and output
#-------------------------------------------------------------------------------
# simulations for each rep
if rep is not None :
    indir   = os.path.join (rootdir, str (seed), protein, 'rep_' + str (rep))
else :
    indir  = os.path.join (rootdir, str (seed), protein)

# outdir
outdir = indir
# input data
datadir = os.path.join (rootdir, str (seed), protein)

print ('input directory for D')
print (indir)

# observed D matrix
Rmat  = np.loadtxt (os.path.join (indir, protein + '_Rmat.txt'))
Lmat  = np.loadtxt (os.path.join (indir, protein + '_Lmat.txt'))
Dmat  = np.loadtxt (os.path.join (indir, protein + '_Dmat.txt'))
LDmat = np.loadtxt (os.path.join (indir, protein + '_LDmat.txt'))


# ligand x protein variants
#-------------------
# singles
Y_singles_mat = np.loadtxt (os.path.join (datadir, protein + '_Y_singles.txt'))
if rep is not None :
    Y_singles = Y_singles_mat[:,rep]
else :
    Y_singles = Y_singles_mat

ligand_singles_mat = np.loadtxt (os.path.join (datadir, ligand + '_Y_singles.txt'))
if rep is not None :
    Y_ligand = ligand_singles_mat[:,rep]
else :
    Y_ligand = ligand_singles_mat


if combine :
    Y_singles = np.nanmean (np.log10 (Y_singles), axis=1)
    Y_ligand  = np.nanmean (np.log10 (Y_ligand), axis=1)

# pickle the simulated read data
filename   = os.path.join (indir, protein + '_Dmats.pkl')
fileObject = open (filename, 'rb')
Dmats_sim = pickle.load (fileObject)
fileObject.close ()


# Compute the p-values 
#-------------------------------------------------------------------------------
# compute the rank of the singles
order_ys  = compute_mean_ranks (Y_singles)
order_lig = compute_mean_ranks (Y_ligand)

print (np.sum (~np.isnan (order_ys)))
print (np.nanmax (order_ys))
print (Dmats_sim.shape)

# compute p-values
P = compute_p_values_trans (D_obs=Dmat, D_sim=Dmats_sim,
                            row_rank=order_lig, col_rank=order_ys)


# save
np.savetxt (os.path.join (indir, protein + '_P.txt'), P)


# Protein x ligand 
#-------------------------------------------------------------------------------
# pickle the simulated read data
filename   = os.path.join (indir, protein + '_LDmats.pkl')
fileObject = open (filename, 'rb')
LDmats_sim = pickle.load (fileObject)
fileObject.close ()

print (LDmat.shape)
print (Dmats_sim.shape)

# compute p-values
Plig = compute_p_values_trans (D_obs=LDmat, D_sim=LDmats_sim,
                               row_rank=order_ys, col_rank=order_lig)


# save
np.savetxt (os.path.join (indir, protein + '_Pligand.txt'), Plig)


del Plig

# Symmetric 
#-------------------------------------------------------------------------------
# compute p-values
Psym = compute_p_values_trans_symmetric (D_obs=Dmat, D_obs_t=LDmat,
                                         D_sim=Dmats_sim, D_sim_t=LDmats_sim,
                                         row_rank=order_lig, col_rank=order_ys)


# save
np.savetxt (os.path.join (indir, protein + '_Psymmetric.txt'), Psym)

del Psym

# Two-sided 
#-------------------------------------------------------------------------------
# compute p-values
Ptwo = compute_p_values_trans_two_sided (D_obs=Dmat, D_obs_t=LDmat,
                                         D_sim=Dmats_sim, D_sim_t=LDmats_sim,
                                         row_rank=order_lig, col_rank=order_ys)


# save
np.savetxt (os.path.join (indir, protein + '_Ptwosided.txt'), Ptwo)

del Ptwo

# Symmetric + scaled
#-------------------------------------------------------------------------------
# find appropriate scaling factors
a_val = np.max (np.sum (~np.isnan (Dmat), axis=1))
b_val = np.max (np.sum (~np.isnan (LDmat), axis=1))

# compute p-values
Pscaled = compute_p_values_trans_symmetric (D_obs=Dmat*(b_val / a_val),
                                            D_obs_t=LDmat,
                                            D_sim=Dmats_sim*(b_val / a_val),
                                            D_sim_t=LDmats_sim,
                                            row_rank=order_lig, col_rank=order_ys)


# save
np.savetxt (os.path.join (indir, protein + '_Psymmetric_scaled.txt'), Pscaled)

del Pscaled

# INTEGRATING
# pickle the simulated read data
if integrate :
    filename   = os.path.join (indir, protein + '_Dmats_unsorted.pkl')
    fileObject = open (filename, 'rb')
    Dmats_sim_unsorted = pickle.load (fileObject)
    fileObject.close ()
    
    # pickle the simulated read data
    filename   = os.path.join (indir, protein + '_LDmats_unsorted.pkl')
    fileObject = open (filename, 'rb')
    LDmats_sim_unsorted = pickle.load (fileObject)
    fileObject.close ()
    
    # compute p-values
    Pint = compute_p_values_trans_integrate (D_obs=Dmat, D_obs_t=LDmat,
                                             D_sim=Dmats_sim, D_sim_t=LDmats_sim,
                                             D_sim_u=Dmats_sim_unsorted, D_sim_ut=LDmats_sim_unsorted,
                                             row_rank=order_lig, col_rank=order_ys)
    
    # save
    np.savetxt (os.path.join (indir, protein + '_P_integrate.txt'), Pint)
