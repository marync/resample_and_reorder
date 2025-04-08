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

import format_helper_functions
import rank_functions

from format_helper_functions import *
from rank_functions import *


# Hardcoded parameters
#-------------------------------------------------------------------------------
ncores  = int (sys.argv[1])
nsims   = int (sys.argv[2])
rootdir = str (sys.argv[3])
seed    = int (sys.argv[4])
protein = str (sys.argv[5])

if len (sys.argv) == 7 :
    rep = int (sys.argv[-1])
else :
    rep = None

# number of amino acids
#datadir = os.path.join ('data', protein)

# Seed random number generators
#-------------------------------------------------------------------------------
# seed an RNG for each simulation run
ss          = np.random.SeedSequence (seed)
child_seeds = ss.spawn (nsims)
streams     = [np.random.default_rng (s) for s in child_seeds]


# Input and output
#-------------------------------------------------------------------------------
#today  = ('_').join (str (datetime.date.today ()).split ('-'))
if rep is not None :
    indir = os.path.join (rootdir, str (seed), 'rep_' + str (rep))
else :
    indir = os.path.join (rootdir, str (seed))

outdir = indir

# read in real data (Dmat)
Rmat = np.loadtxt (os.path.join (indir, protein + '_Rmat.txt'))

# read in simulated data
dreadsfile = open (os.path.join (indir, protein + '_DoubleSimYs.pkl'), 'rb')
simDoubYs  = pickle.load (dreadsfile)
sreadsfile = open (os.path.join (indir, protein + '_SingleSimYs.pkl'), 'rb')
simSingYs  = pickle.load (sreadsfile)

# number site-AA combos and number of positions
L = Rmat.shape[0]

# number to scale to
M = np.nanmax ( Rmat )
print (M)

# Compute the rank matrices
Dlist = Parallel (n_jobs=ncores) (delayed (compute_transformed_Rmat)
                                     (simDoubYs[:,:,i], M=M) for i in range (nsims))
Slist = Parallel (n_jobs=ncores) (delayed (compute_rank_sample)
                                     (simSingYs[:,i], rng=streams[i]) for i in range (nsims))

# put in matrices
Rmats  = np.zeros ((nsims, L, L))
Rsings = np.zeros ((nsims, L))
for i in range (nsims) :
    Rmats[i,:,:] = Dlist[i]
    Rsings[i,:]  = Slist[i]

# pickle the simulated read data
filename   = os.path.join (outdir, protein + '_Rmats_doubles.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Rmats, fileObject)
fileObject.close ()

filename   = os.path.join (outdir, protein + '_Rs_singles.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Rsings, fileObject)
fileObject.close ()

