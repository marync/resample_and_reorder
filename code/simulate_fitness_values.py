import os
import sys
import copy as cp
import pandas
import scipy
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle

# my functions
sys.path.insert(1, '../code/')

import bootstrapping_functions
from bootstrapping_functions import *

# Hardcoded parameters
#-------------------------------------------------------------------------------
protein   = str (sys.argv[1])
nsims     = int (sys.argv[2])
rootout   = str (sys.argv[3])
seed      = int (sys.argv[4])
readthres = int (sys.argv[5])

indir = None
if len (sys.argv) == 7 :
    indir = str (sys.argv[6])
    print (indir)

combine     = False # str (sys.argv[6]) == 'True'
simulate_N0 = True  # str (sys.argv[6]) == 'True' # whether to simulate the initial reads
eta         = 1 # pseudo-count
inside      = True

# in directory
rootin = '.'

# how missing data is encoded in read matrices
if protein in ['fosjun','GB1'] :
    NA = -1
else :
    NA = None 

# number of amino acids
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
outdir = rootout
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

outdir = os.path.join (outdir, str (seed))
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

if indir is None :
    indir = outdir

print ('Input directory: ' + str (indir))

# Read in data
#-------------------------------------------------------------------------------
s_neu = np.loadtxt (os.path.join (indir, protein + '_S_neutral.txt'))
s_sel = np.loadtxt (os.path.join (indir, protein + '_S_selection.txt'))

if protein in ['fosjun','fos','jun'] :
    d_neu = pickle.load ( open (os.path.join (indir, protein + '_D_neutral.pkl'), 'rb') )
    y_doubles = pickle.load ( open (os.path.join (indir, protein + '_Y_doubles.pkl'), 'rb') )
else : 
    d_neu = np.loadtxt (os.path.join (indir, protein + '_D_neutral.txt'))
    y_doubles = np.loadtxt (os.path.join (indir, protein + '_Y_doubles.txt'))

# fitness estimates
y_singles = np.loadtxt (os.path.join (indir, protein + '_Y_singles.txt'))

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

   
# Compute ranks for singles and doubles for each replicate
#-------------------------------------------------------------------------------
for i in range (n_reps) :

    if n_reps > 1 and not combine :
        out_i = os.path.join (outdir, 'rep_' + str (i) )
    else :
        out_i = outdir
        
    if not os.path.isdir (out_i) :
        os.makedirs (out_i)
    
    if n_reps > 1 :
        SS = bootstrap_fitness_singles (reads_neu_sing=s_neu[:,i],
                                        reads_sel_sing=s_sel[:,i], N0=simulate_N0,
                                        nsims=nsims, rng=rng, inside=inside,
                                        eta=eta, NA=NA, threshold=readthres)
        SD = bootstrap_fitness_doubles (reads_neu_doub=d_neu[:,:,i],
                                        reads_sel_doub=d_sel[:,:,i], N0=simulate_N0,
                                        nsims=nsims, rng=rng, inside=inside,
                                        eta=eta, NA=NA, threshold=readthres)
    
        if combine :
            print ('combine')
            print (SS.shape)
            print (SD.shape)

    else :
        SS = bootstrap_fitness_singles (reads_neu_sing=s_neu,
                                        reads_sel_sing=s_sel, N0=simulate_N0,
                                        nsims=nsims, rng=rng, inside=inside,
                                        eta=eta, NA=NA, threshold=readthres)
        SD = bootstrap_fitness_doubles (reads_neu_doub=d_neu,
                                        reads_sel_doub=d_sel, N0=simulate_N0,
                                        nsims=nsims, rng=rng, inside=inside,
                                        eta=eta, NA=NA, threshold=readthres)

    # pickle the simulated read data
    filename   = os.path.join (out_i, protein + '_SingleSimYs.pkl')
    fileObject = open (filename, 'wb')
    pickle.dump (SS, fileObject)
    fileObject.close ()

    filename   = os.path.join (out_i, protein + '_DoubleSimYs.pkl')
    fileObject = open (filename, 'wb')
    pickle.dump (SD, fileObject)
    fileObject.close ()


