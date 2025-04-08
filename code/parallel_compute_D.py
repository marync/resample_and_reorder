import os
import sys
import copy as cp
import numpy as np
import datetime
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pickle

# my functions
sys.path.insert(1, '../code/')

from D_functions import *

# Hardcoded parameters
#-------------------------------------------------------------------------------
ncores  = int (sys.argv[1])
rootdir = str (sys.argv[2])
seed    = int (sys.argv[3])
protein = str (sys.argv[4])

if len (sys.argv) == 6 :
    rep = int (sys.argv[-1])
else :
    rep = None

# number of amino acids
k_impute = 5 # how far to look for a nn when imputing D matrix


# Input and output
#-------------------------------------------------------------------------------
if rep is not None :
    indir = os.path.join (rootdir, str (seed), 'rep_' + str (rep))
else :
    indir = os.path.join (rootdir, str (seed))
outdir = indir

#today  = ('_').join (str (datetime.date.today ()).split ('-'))
#if rep is not None :
#    indir = os.path.join (rootdir, today, str (seed), 'rep_' + str (rep))
#else :
#    indir = os.path.join (rootdir, today, str (seed))
outdir = indir

print ('input directory for D')
print (indir)

# observed rank matrix
Rmat = np.loadtxt (os.path.join (indir, protein + '_Rmat.txt'))

# read in simulated data
dreadsfile = open (os.path.join (indir, protein + '_DoubleSimYs.pkl'), 'rb')
simDoubYs  = pickle.load (dreadsfile)
dreadsfile.close ()

# rank matrices
rfile = open (os.path.join (indir, protein + '_Rmats_doubles.pkl'), 'rb')
Rmats_doubles = pickle.load ( rfile )
rfile.close ()

sreadsfile = open (os.path.join (indir, protein + '_SingleSimYs.pkl'), 'rb')
simSingYs  = pickle.load (sreadsfile)
sreadsfile.close ()

# number site-AA combos and number of positions
L, nsims = simSingYs.shape

if protein == 'fosjun' :
    nAA = 21
else :
    nAA = 20
#if protein in ['GB1', 'sim'] :

# Seed random number generators
#-------------------------------------------------------------------------------
# seed an RNG for each simulation run
ss          = np.random.SeedSequence (seed)
child_seeds = ss.spawn (nsims)
streams     = [np.random.default_rng (s) for s in child_seeds]


# Compute the rank matrices 
#-------------------------------------------------------------------------------
# number to scale to
M  = int (np.nanmax ( Rmat )) # max double rank
if protein == 'fosjun' :
    Ms = 2*M + 1
    dprot = protein
else :
    Ms = np.max (np.sum (~np.isnan (simSingYs), axis=0)) - 1
    #Ms = int (M + (nAA-1))        # max single rank
    dprot = None
print ()
print ('M: ' + str (M))
print ('Ms: ' + str (Ms))
print ()

# Compute rank matrices unsorted
Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
                                     (singles=simSingYs[:,i],
                                      Rmat=Rmats_doubles[i,:,:],
                                      protein=dprot,
                                      rng=None, sort=False,
                                      M=M, nAA=nAA) for i in range (nsims))

# put in matrices
Dmats = np.zeros ((nsims, L, L))
for i in range (nsims) :
    Dmats[i,:,:] = Dlist[i] 


# pickle the simulated read data
filename   = os.path.join (outdir, protein + '_Dmats_unsorted.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Dmats, fileObject)
fileObject.close ()

del Dlist
del Dmats


# Compute the rank matrices
Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
                                     (singles=simSingYs[:,i],
                                      Rmat=Rmats_doubles[i,:,:],
                                      protein=dprot,
                                      rng=None, sort=True,
                                      M=M, nAA=nAA) for i in range (nsims))

# write Dmats to file 
#Dmats_raw = np.zeros_like (Rmats_doubles)
#for i in range (nsims) :
#    Dmats_raw[i,:,:] = Dlist[i]

#filename   = os.path.join (outdir, protein + '_Dmats_raw_' + stat + '.pkl')
#fileObject = open (filename, 'wb')
#pickle.dump (Dmats_raw, fileObject)
#fileObject.close ()
#del Dmats_raw


Dlist_imputed = Parallel (n_jobs=ncores) (delayed (impute_matrix_nn)
                                                (A=Dlist[i][:(Ms+1),:(Ms+1)], k=k_impute,
                                                 rng=streams[i])
                                                for i in range (nsims)
                                            )

del Dlist

# put in matrices
Dmats_imp = np.zeros ((nsims, Ms+1, Ms+1))
for i in range (nsims) :
    Dmats_imp[i,:,:] = Dlist_imputed[i] 


# pickle the simulated read data
filename   = os.path.join (outdir, protein + '_Dmats.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Dmats_imp, fileObject)
fileObject.close ()

#filename   = os.path.join (outdir, protein + '_Dmats_raw_' + stat + '.pkl')
#fileObject = open (filename, 'wb')
#pickle.dump (Dmats_raw, fileObject)
#fileObject.close ()

