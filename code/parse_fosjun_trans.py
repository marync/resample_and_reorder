import os
import sys
import numpy as np
import pandas
import pickle

from format_helper_functions import *

# read count threshold
readthres = int (sys.argv[1])
rootout   = sys.argv[2]
protein   = sys.argv[3]
seed      = int (sys.argv[4])

# files
indir  = os.path.join ('data', 'fosjun')

outdir = rootout
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

outdir = os.path.join (outdir, str (seed))
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

outdir = os.path.join (outdir, protein)
if not os.path.isdir (outdir) :
    os.makedirs (outdir)


neutral_files = ['SRR5952429_counts.tsv', 'SRR5952430_counts.tsv', 'SRR5952431_counts.tsv']
select_files  = ['SRR5952432_counts.tsv', 'SRR5952433_counts.tsv', 'SRR5952434_counts.tsv']

# replicates
n_sel = 3
n_neu = 3
nreps = 3
nAA   = 21
nfos  = 32
npos  = 64
eta   = 1
Lfos  = nAA*nfos

# AA to numeric dictionary
aadict = make_aa_dict (stop='*')

# process data
D_neu = np.zeros ((npos*nAA, npos*nAA, n_neu), dtype=int) * -1
S_neu = np.zeros ( (npos*nAA, n_neu) ) * -1

ct = 0
for nf in neutral_files :
    print (nf)
    df = pandas.read_csv ( os.path.join (indir, nf), delimiter='\t' )
    for index, row in df.iterrows () :
        if row['Fos_mut'] != '_wt' and row['Jun_mut'] != '_wt' :
            pos_1 = int (row['Fos_mut'][1:-1]) - 1
            pos_2 = int (row['Jun_mut'][1:-1]) - 1 + nfos
                    
            aa_1  = row['Fos_mut'][-1]
            aa_2  = row['Jun_mut'][-1]
    
            D_neu[pos_1*nAA + aadict[aa_1], pos_2*nAA + aadict[aa_2], ct] = row['count']
            D_neu[pos_2*nAA + aadict[aa_2], pos_1*nAA + aadict[aa_1], ct] = row['count']
            
        if row['Fos_mut'] == '_wt' and row['Jun_mut'] != '_wt':
            pos = int (row['Jun_mut'][1:-1]) - 1 + nfos
            aa  = row['Jun_mut'][-1]
    
            S_neu[pos*nAA + aadict[aa], ct] = row['count']
        
        if row['Jun_mut'] == '_wt' and row['Fos_mut'] != '_wt' :
            pos = int (row['Fos_mut'][1:-1]) - 1
            aa  = row['Fos_mut'][-1]
    
            S_neu[pos*nAA + aadict[aa], ct] = row['count']
  
    ct += 1

# selection
D_sel = np.zeros ((npos*nAA, npos*nAA, n_sel), dtype=int) * -1 
S_sel = np.zeros ( (npos*nAA, n_sel), dtype=int ) * -1 

ct = 0
for sf in select_files :
    print (sf)
    df = pandas.read_csv ( os.path.join (indir, sf), delimiter='\t' )
    for index, row in df.iterrows () :
        if row['Fos_mut'] != '_wt' and row['Jun_mut'] != '_wt' :
            pos_1 = int (row['Fos_mut'][1:-1]) - 1
            pos_2 = int (row['Jun_mut'][1:-1]) - 1 + nfos
                    
            aa_1  = row['Fos_mut'][-1]
            aa_2  = row['Jun_mut'][-1]
    
            D_sel[pos_1*nAA + aadict[aa_1], pos_2*nAA + aadict[aa_2], ct] = row['count']
            D_sel[pos_2*nAA + aadict[aa_2], pos_1*nAA + aadict[aa_1], ct] = row['count']
            
        if row['Fos_mut'] == '_wt' and row['Jun_mut'] != '_wt':
            pos = int (row['Jun_mut'][1:-1]) - 1 + nfos
            aa  = row['Jun_mut'][-1]
    
            S_sel[pos*nAA + aadict[aa], ct] = row['count']
        
        if row['Jun_mut'] == '_wt' and row['Fos_mut'] != '_wt' :
            pos = int (row['Fos_mut'][1:-1]) - 1
            aa  = row['Fos_mut'][-1]
    
            S_sel[pos*nAA + aadict[aa], ct] = row['count']        

    ct += 1


# filter reads based on read count threshold
D_neu[D_neu < readthres] = -1
D_sel[D_neu == -1]       = -1
S_neu[S_neu < readthres] = -1
S_sel[S_neu == -1]       = -1

# save all files
np.savetxt (os.path.join (outdir, 'fos_S_neutral.txt'), S_neu[:Lfos,:], fmt='%d')
np.savetxt (os.path.join (outdir, 'fos_S_selection.txt'), S_sel[:Lfos,:], fmt='%d')
np.savetxt (os.path.join (outdir, 'jun_S_neutral.txt'), S_neu[Lfos:,:], fmt='%d')
np.savetxt (os.path.join (outdir, 'jun_S_selection.txt'), S_sel[Lfos:,:], fmt='%d')

# save the double info
f = open (os.path.join (outdir, 'fos_D_neutral.pkl'), 'wb')
pickle.dump (D_neu[Lfos:,:,:][:,:Lfos,:], f)
f.close ()

f = open (os.path.join (outdir, 'fos_D_selection.pkl'), 'wb')
pickle.dump ( D_sel[Lfos:,:,:][:,:Lfos,:], f )
f.close ()

# add pseudo-counts
D_neu[D_neu != -1] += eta
D_sel[D_sel != -1] += eta
S_neu[S_neu != -1] += eta
S_sel[S_sel != -1] += eta

# compute fitness estimates for each replicate
Y_singles     = np.zeros_like (S_sel) * np.nan
Y_singles_eta = np.zeros_like (S_sel) * np.nan
for i in range (n_sel) :

    sel_i = np.array (S_sel[:,i], dtype=float)
    neu_i = np.array (S_neu[:,i], dtype=float)
    neu_i[S_neu[:,i] == -1] = np.nan
    neu_i[S_neu[:,i] == 0]  = np.nan

    Y_singles[:,i]     = (sel_i / neu_i)
    #Y_singles_eta[:,i] = (sel_i + eta) / (neu_i + eta)

np.savetxt (os.path.join (outdir, 'fos_Y_singles.txt'), Y_singles[:Lfos])
np.savetxt (os.path.join (outdir, 'jun_Y_singles.txt'), Y_singles[Lfos:])
#np.savetxt (os.path.join (outdir, 'fosjun_Y_singles.txt'), Y_singles_eta)

# doubles
Y_doubles     = np.zeros_like (D_sel) * np.nan
Y_doubles_eta = np.zeros_like (D_sel) * np.nan
for i in range (n_sel) :
    d_sel_f = np.array (D_sel[:,:,i], dtype=float)

    # change to nan
    d_neu_f = np.array (D_neu[:,:,i], dtype=float)
    d_neu_f[D_neu[:,:,i] == -1] = np.nan
    d_neu_f[D_neu[:,:,i] == 0]  = np.nan

    Y_doubles[:,:,i]     = (d_sel_f / d_neu_f)
    #Y_doubles_eta[:,:,i] = (d_sel_f + eta) / (d_neu_f + eta)

pickle.dump (Y_doubles[Lfos:,:,:][:,:Lfos,:], open (os.path.join (outdir, 'fos_Y_doubles.pkl'), 'wb'))
#pickle.dump (Y_doubles_eta, open (os.path.join (outdir, 'fosjun_Y_doubles.pkl'), 'wb'))



