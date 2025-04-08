import os
import sys
import copy as cp
import numpy as np
import pickle

# my functions
sys.path.insert(1, '../code/')

import bootstrapping_functions
from bootstrapping_functions import *

# Hardcoded parameters
#-------------------------------------------------------------------------------
protdir   = str (sys.argv[1])
protein   = str (sys.argv[2])
ligand    = str (sys.argv[3])
readthres = int (sys.argv[4])
nsims     = int (sys.argv[5])
rootout   = str (sys.argv[6])
seed      = int (sys.argv[7])

# pseudo-count
eta    = 1
#inside = False
inside = True
if protein == 'fos' :
    NA = -1
else :
    NA = None 

# Seed random number generators
#-------------------------------------------------------------------------------
# only need one RNG for sampling reads
rng = np.random.default_rng (seed)

# Input and output
#-------------------------------------------------------------------------------
outdir = rootout
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

outdir = os.path.join (outdir, str (seed))
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

outdir = os.path.join (outdir, protein)
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

indir = outdir

# Read in data
#-------------------------------------------------------------------------------
s_neu = np.loadtxt (os.path.join (indir, protein + '_S_neutral.txt'))
s_sel = np.loadtxt (os.path.join (indir, protein + '_S_selection.txt'))

# ligand
s_neu_lig = np.loadtxt (os.path.join (indir, ligand + '_S_neutral.txt'))
s_sel_lig = np.loadtxt (os.path.join (indir, ligand + '_S_selection.txt'))

# doubles
d_neu = pickle.load ( open (os.path.join (indir, protein + '_D_neutral.pkl'), 'rb') )
d_sel = pickle.load ( open (os.path.join (indir, protein + '_D_selection.pkl'), 'rb') )
y_doubles = pickle.load ( open (os.path.join (indir, protein + '_Y_doubles.pkl'), 'rb') )

# fitness estimates
y_singles     = np.loadtxt (os.path.join (indir, protein + '_Y_singles.txt'))
y_singles_lig = np.loadtxt (os.path.join (indir, ligand  + '_Y_singles.txt'))

# position info
L, n_reps     = y_singles.shape      # protein
L_lig, n_reps = y_singles_lig.shape  # ligand


# Compute ranks for singles and doubles for each replicate
#-------------------------------------------------------------------------------
for i in range (n_reps) :

    out_i = os.path.join (outdir, 'rep_' + str (i) )
    if not os.path.isdir (out_i) :
        os.makedirs (out_i)

    SS    = bootstrap_fitness_singles (reads_neu_sing=s_neu[:,i],
                                       reads_sel_sing=s_sel[:,i],
                                       nsims=nsims, rng=rng, inside=inside,
                                       eta=eta, NA=NA) #threshold=readthres)
    SSlig = bootstrap_fitness_singles (reads_neu_sing=s_neu_lig[:,i],
                                       reads_sel_sing=s_sel_lig[:,i],
                                       nsims=nsims, rng=rng, inside=inside,
                                       eta=eta, NA=NA, threshold=readthres)
    SD = bootstrap_fitness_doubles_trans (reads_neu_doub=d_neu[:,:,i],
                                          reads_sel_doub=d_sel[:,:,i],
                                          nsims=nsims, rng=rng, inside=inside,
                                          eta=eta, NA=NA, threshold=readthres)

    # pickle the simulated read data
    filename   = os.path.join (out_i, protein + '_SingleSimYs.pkl')
    fileObject = open (filename, 'wb')
    pickle.dump (SS, fileObject)
    fileObject.close ()
    
    filename   = os.path.join (out_i, ligand + '_SingleSimYs.pkl')
    fileObject = open (filename, 'wb')
    pickle.dump (SSlig, fileObject)
    fileObject.close ()

    filename   = os.path.join (out_i, protein + '_DoubleSimYs.pkl')
    fileObject = open (filename, 'wb')
    pickle.dump (SD, fileObject)
    fileObject.close ()


print ()
print (SS.shape)
print (SSlig.shape)
print (SD.shape)
print ()
