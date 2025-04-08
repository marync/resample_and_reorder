import os
import sys
import copy as cp
import numpy as np
from joblib import Parallel, delayed
import pickle

# my functions
sys.path.insert(1, '../code/')

import rank_functions
from rank_functions import *


# Hardcoded parameters
#-------------------------------------------------------------------------------
protein = str (sys.argv[1])
ligand  = str (sys.argv[2])
rootdir = str (sys.argv[3])
seed    = int (sys.argv[4])
ncores  = int (sys.argv[5])

if len (sys.argv) == 7 :
    rep = int (sys.argv[6])
else :
    rep = None

#sims = True
sims = False

# Input and output
#-------------------------------------------------------------------------------
#today  = ('_').join (str (datetime.date.today ()).split ('-'))
if rep is not None :
    indir  = os.path.join (rootdir, str (seed), protein, 'rep_' + str (rep))
    outdir = indir
else :
    indir  = os.path.join (rootdir, str (seed), protein)
    outdir = indir

# read in real data (Dmat)
Rmat = np.loadtxt (os.path.join (indir, protein + '_Rmat.txt'))
Lmat = np.loadtxt (os.path.join (indir, protein + '_Lmat.txt'))

# read in simulated data
dreadsfile = open (os.path.join (indir, protein + '_DoubleSimYs.pkl'), 'rb')
simDoubYs  = pickle.load (dreadsfile)
dreadsfile.close ()

sreadsfile = open (os.path.join (indir, protein + '_SingleSimYs.pkl'), 'rb')
simSingYs  = pickle.load (sreadsfile)
sreadsfile.close ()

lreadsfile = open (os.path.join (indir, ligand + '_SingleSimYs.pkl'), 'rb')
simSingLig = pickle.load (lreadsfile)
lreadsfile.close ()

# number ligand, number protein, number of simulations
if sims :
    nsims, L_lig, L = simDoubYs.shape
else :
    L_lig, L, nsims = simDoubYs.shape


print (simSingYs.shape)
print (simSingLig.shape)
print (simDoubYs.shape)

# Seed random number generators
#-------------------------------------------------------------------------------
# seed an RNG for each simulation run
ss          = np.random.SeedSequence (seed)
child_seeds = ss.spawn (nsims)
streams     = [np.random.default_rng (s) for s in child_seeds]


# Computation - ligand x pdz
#-------------------------------------------------------------------------------
# number to scale to
M = np.nanmax ( Rmat )
print (M)

# Compute the rank matrices
if sims :
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_transformed_Rmat)
                                     (simDoubYs[i,:,:], M=M) for i in range (nsims))
else :
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_transformed_Rmat)
                                     (simDoubYs[:,:,i], M=M) for i in range (nsims))

# store together
Rmats = np.zeros ((nsims, L_lig, L))
for i in range (nsims) :
    Rmats[i,:,:] = Dlist[i]

# pickle the simulated read data
filename   = os.path.join (outdir, protein + '_Rmats_doubles.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Rmats, fileObject)
fileObject.close ()

del Dlist
del Rmats

if sims :
    Slist = Parallel (n_jobs=ncores) (delayed (compute_rank_sample)
                                      (simSingYs[i,:], rng=streams[i]) for i in range (nsims))
    Liglist = Parallel (n_jobs=ncores) (delayed (compute_rank_sample)
                                       (simSingLig[i,:], rng=streams[i]) for i in range (nsims))
else :
    Slist = Parallel (n_jobs=ncores) (delayed (compute_rank_sample)
                                     (simSingYs[:,i], rng=streams[i]) for i in range (nsims))
    Liglist = Parallel (n_jobs=ncores) (delayed (compute_rank_sample)
                                     (simSingLig[:,i], rng=streams[i]) for i in range (nsims))

# put in matrices
Rsings  = np.zeros ((nsims, L))
Rligand = np.zeros ((nsims, L_lig))
for i in range (nsims) :
    Rsings[i,:]  = Slist[i]
    Rligand[i,:] = Liglist[i]

filename   = os.path.join (outdir, protein + '_Rs_singles.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Rsings, fileObject)
fileObject.close ()

filename   = os.path.join (outdir, ligand + '_Rs_singles.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Rligand, fileObject)
fileObject.close ()

del Slist
del Liglist
del Rsings
del Rligand

# Computation - pdz x ligand
#-------------------------------------------------------------------------------
# number to scale to
M = np.nanmax ( Lmat )
print (M)

# Compute the rank matrices
if sims :
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_transformed_Rmat)
                                     (np.transpose (simDoubYs[i,:,:]), M=M) for i in range (nsims))
else :
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_transformed_Rmat)
                                     (np.transpose (simDoubYs[:,:,i]), M=M) for i in range (nsims))

# store together
Rmats = np.zeros ((nsims, L, L_lig))
for i in range (nsims) :
    Rmats[i,:,:] = Dlist[i]

# pickle the simulated read data
filename   = os.path.join (outdir, protein + '_Lmats_doubles.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Rmats, fileObject)
fileObject.close ()
