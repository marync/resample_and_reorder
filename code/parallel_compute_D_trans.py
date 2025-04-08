import os
import sys
import copy as cp
import numpy as np
from joblib import Parallel, delayed
import pickle

# my functions
sys.path.insert(1, '../code/')

from D_functions import *

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

# number of amino acids
k_impute = 5 # how far to look for a nn when imputing D matrix
nAA = 21

#normal = True
normal = False

# Input and output
#-------------------------------------------------------------------------------
#today  = ('_').join (str (datetime.date.today ()).split ('-'))
if rep is not None :
    datadir = os.path.join (rootdir, str (seed), protein) 
    indir   = os.path.join (rootdir, str (seed), protein, 'rep_' + str (rep))
    outdir  = indir
else :
    indir  = os.path.join (rootdir, str (seed), protein)
    outdir = indir

# observed rank matrix
Rmat = np.loadtxt (os.path.join (indir, protein + '_Rmat.txt'))
Lmat = np.loadtxt (os.path.join (indir, protein + '_Lmat.txt'))
Yp   = np.loadtxt (os.path.join (datadir, protein + '_Y_singles.txt'))
Ylig = np.loadtxt (os.path.join (datadir, ligand + '_Y_singles.txt'))

# read in simulated data
dreadsfile = open (os.path.join (indir, protein + '_DoubleSimYs.pkl'), 'rb')
simDoubYs  = pickle.load (dreadsfile)
dreadsfile.close ()

# rank matrices
rfile = open (os.path.join (indir, protein + '_Rmats_doubles.pkl'), 'rb')
Rmats_doubles = pickle.load ( rfile )
rfile.close ()

# single proteins
sreadsfile = open (os.path.join (indir, protein + '_SingleSimYs.pkl'), 'rb')
simSingYs  = pickle.load (sreadsfile)
sreadsfile.close ()

# single ligands
lreadsfile = open (os.path.join (indir, ligand + '_SingleSimYs.pkl'), 'rb')
simSingLig = pickle.load (lreadsfile)
lreadsfile.close ()

# number site-AA combos and number of positions
if normal :
    nsims, L_lig, L = simDoubYs.shape 
else :
    L_lig, L, nsims = simDoubYs.shape 


# Seed random number generators
#-------------------------------------------------------------------------------
# seed an RNG for each simulation run
ss          = np.random.SeedSequence (seed)
child_seeds = ss.spawn (nsims)
streams     = [np.random.default_rng (s) for s in child_seeds]


# Compute the rank matrices 
#-------------------------------------------------------------------------------
# number to scale to
M    = int (np.nanmax ( Rmat )) # max double rank
if normal :
    Mp   = np.sum (~np.isnan (Yp))
    Mlig = np.sum (~np.isnan (Ylig))
else :
    # number for single protein variants
    Mp = np.max (np.sum (~np.isnan (simSingYs), axis=0)) - 1
    # number for ligands
    Mlig = np.max (np.sum (~np.isnan (simSingLig), axis=0)) - 1

print ()
print ('M: ' + str (M))
print ('Mp: ' + str (Mp))
print ('Mlig: ' + str (Mlig))
print ()

if normal :
    # Compute the rank matrices
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
            (singles=simSingYs[i,:],
                                           Rmat=Rmats_doubles[i,:,:],
                                           ligand=simSingLig[i,:],
                                           rng=streams[i],
                                           protein='trans',
                                           sort=False,
                                           M=M, nAA=nAA) for i in range (nsims))
else :
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
                                          (singles=simSingYs[:,i],
                                           Rmat=Rmats_doubles[i,:,:],
                                           ligand=simSingLig[:,i],
                                           rng=streams[i],
                                           protein='trans',
                                           sort=False,
                                           M=M, nAA=nAA) for i in range (nsims))

# put in matrices
Dmats = np.zeros ((nsims, L_lig, L))
for i in range (nsims) :
    Dmats[i,:,:] = Dlist[i]

# pickle the simulated read data
filename   = os.path.join (outdir, protein + '_Dmats_unsorted.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Dmats, fileObject)
fileObject.close ()

del Dlist
del Dmats


if normal :
# Compute the rank matrices
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
                                        (singles=simSingYs[i,:],
                                           Rmat=Rmats_doubles[i,:,:],
                                           ligand=simSingLig[i,:],
                                           rng=streams[i],
                                           protein='trans',
                                           sort=True,
                                           M=M, nAA=nAA) for i in range (nsims))
else :
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
                                          (singles=simSingYs[:,i],
                                           Rmat=Rmats_doubles[i,:,:],
                                           ligand=simSingLig[:,i],
                                           rng=streams[i],
                                           protein='trans',
                                           sort=True,
                                           M=M, nAA=nAA) for i in range (nsims))


Dlist_imputed = Parallel (n_jobs=ncores) (delayed (impute_matrix_nn)
                                                (A=Dlist[i][:(Mlig+1),:(Mp+1)], k=k_impute,
                                                 rng=streams[i])
                                                for i in range (nsims)
                                            )


# put in matrices
Dmats_imp = np.zeros ((nsims, Mlig+1, Mp+1))
Dmats_raw = np.zeros_like (Rmats_doubles)
for i in range (nsims) :
    Dmats_imp[i,:,:] = Dlist_imputed[i] 
    Dmats_raw[i,:,:] = Dlist[i]

print ('SHAPE:')
print (Mp)
print (Mlig)
print (Dmats_imp.shape)


# pickle the simulated read data
filename   = os.path.join (outdir, protein + '_Dmats.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Dmats_imp, fileObject)
fileObject.close ()

filename   = os.path.join (outdir, protein + '_Dmats_raw.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Dmats_raw, fileObject)
fileObject.close ()

del Rmats_doubles
del Dlist
del Dlist_imputed
del Dmats_imp
del Dmats_raw


# Compute the rank matrices for protein x ligand 
#-------------------------------------------------------------------------------
# rank matrices
rfile = open (os.path.join (indir, protein + '_Lmats_doubles.pkl'), 'rb')
Rmats_doubles = pickle.load ( rfile )
rfile.close ()

# number to scale to
M  = int (np.nanmax ( Lmat )) # max double rank

print ()
print ('M: ' + str (M))
print ('Mp: ' + str (Mp))
print ('Mlig: ' + str (Mlig))
print ()


if normal :
# Compute the rank matrices
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
            (singles=simSingLig[i,:],
                                           Rmat=Rmats_doubles[i,:,:],
                                           ligand=simSingYs[i,:],
                                           rng=streams[i],
                                           protein='trans',
                                           sort=False,
                                           M=M, nAA=nAA) for i in range (nsims))
else :
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
                                          (singles=simSingLig[:,i],
                                           Rmat=Rmats_doubles[i,:,:],
                                           ligand=simSingYs[:,i],
                                           rng=streams[i],
                                           protein='trans',
                                           sort=False,
                                           M=M, nAA=nAA) for i in range (nsims))

# put in matrices
Dmats = np.zeros ((nsims, L, L_lig))
for i in range (nsims) :
    Dmats[i,:,:] = Dlist[i]

# pickle the simulated read data
filename   = os.path.join (outdir, protein + '_LDmats_unsorted.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Dmats, fileObject)
fileObject.close ()

del Dlist
del Dmats


if normal :
# Compute the rank matrices
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
            (singles=simSingLig[i,:],
                                           Rmat=Rmats_doubles[i,:,:],
                                           ligand=simSingYs[i,:],
                                           rng=streams[i],
                                           protein='trans',
                                           sort=True,
                                           M=M, nAA=nAA) for i in range (nsims))
else :
    Dlist = Parallel (n_jobs=ncores) (delayed (compute_D_from_ranks)
            (singles=simSingLig[:,i],
                                           Rmat=Rmats_doubles[i,:,:],
                                           ligand=simSingYs[:,i],
                                           rng=streams[i],
                                           protein='trans',
                                           sort=True,
                                           M=M, nAA=nAA) for i in range (nsims))


Dlist_imputed = Parallel (n_jobs=ncores) (delayed (impute_matrix_nn)
                                                (A=Dlist[i][:(Mp+1),:(Mlig+1)], k=k_impute,
                                                 rng=streams[i])
                                                for i in range (nsims)
                                            )


# put in matrices
Dmats_imp = np.zeros ((nsims, Mp+1, Mlig+1))
Dmats_raw = np.zeros_like (Rmats_doubles)
for i in range (nsims) :
    Dmats_imp[i,:,:] = Dlist_imputed[i] 
    Dmats_raw[i,:,:] = Dlist[i]


# pickle the simulated read data
filename   = os.path.join (outdir, protein + '_LDmats.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Dmats_imp, fileObject)
fileObject.close ()

filename   = os.path.join (outdir, protein + '_LDmats_raw.pkl')
fileObject = open (filename, 'wb')
pickle.dump (Dmats_raw, fileObject)
fileObject.close ()

