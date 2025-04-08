import os
import sys
import copy as cp
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
outname  = sys.argv[2]
outdir   = sys.argv[3]

Pos = np.loadtxt (filename, delimiter=',')

positions = Pos[:,0]
npos = len (np.unique (positions))
upositions = np.sort (np.unique (positions))

print (upositions)
print (npos)

Dmat = np.zeros ( (npos, npos) )
for i in range (npos) :
    p_i = upositions[i]
    for j in range (i+1, npos) :
        p_j = upositions[j]
        pos_i = cp.deepcopy (Pos[positions == p_i,1:])
        pos_j = cp.deepcopy (Pos[positions == p_j,1:])

        min_ij = np.inf
        for k in range (pos_i.shape[0]) :
            for l in range (pos_j.shape[0]) :   
                dkl = np.sqrt ( np.sum ( (pos_i[k,:] - pos_j[l,:])**2) )
                min_ij = np.min ([min_ij, dkl])

        Dmat[i,j] = Dmat[j,i] = min_ij

np.savetxt (os.path.join (outdir, outname + '_distance_matrix.txt'), Dmat)
