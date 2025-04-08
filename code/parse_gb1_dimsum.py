import os
import sys
import numpy as np
import pandas
import pickle

from format_helper_functions import *

indir   = sys.argv[1]
outdir  = sys.argv[2]

if len (sys.argv) >= 4 :
    protein = sys.argv[3]
else :
    protein = 'GB1'

if len (sys.argv) == 6 :
    input_wt  = [int (sys.argv[4])]
    output_wt = [int (sys.argv[5])]
else :
    input_wt  = [1759616]
    output_wt = [3041819]

# pseudo-count
eta = 1

code_dict = {
        'A': ['gca', 'gcc', 'gcg', 'gct'],
        'C': ['tgt', 'tgc'],
        'D': ['gac', 'gat'],
        'E': ['gag', 'gaa'],
        'F': ['ttt', 'ttc'],
        'G': ['ggt', 'ggg', 'gga', 'ggc'],
        'H': ['cat', 'cac'],
        'I': ['atc', 'ata', 'att'],
        'K': ['aag', 'aaa'],
        'L': ['ctt', 'ctg', 'cta', 'ctc', 'tta', 'ttg'],
        'M': ['atg'],
        'N': ['aac', 'aat'],
        'P': ['cct', 'ccg', 'cca', 'ccc'],
        'Q': ['caa', 'cag'],
        'R': ['agg', 'aga', 'cga', 'cgc', 'cgg', 'cgt'],
        'S': ['agc', 'agt', 'tct', 'tcg', 'tcc', 'tca'],
        'T': ['aca', 'acg', 'act', 'acc'],
        'V': ['gta', 'gtc', 'gtg', 'gtt'],
        'W': ['tgg'],
        'Y': ['tat', 'tac'],
        '*': ['taa', 'tga', 'tag']
    }

def convert_aa_to_nt (seq, code_dict=code_dict) :
    """
    """

    seqlist = list (seq)
    ntlist  = [code_dict[aa][0] for aa in seqlist]
    
    return ('').join (ntlist)


print (outdir)

if not os.path.isdir (outdir) :
    os.makedirs (outdir)

outf  = open (os.path.join (outdir, protein + '_dimsum.txt'), 'w')
outnt = open (os.path.join (outdir, protein + '_dimsum_nt.txt'), 'w')

all_wt = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

# load single reads
N0s_f = os.path.join (indir, protein + '_S_neutral.txt')
N1s_f = os.path.join (indir, protein + '_S_selection.txt')

# load double reads
if protein == 'GB1' :
    N0d = np.loadtxt (os.path.join (indir, protein + '_D_neutral.txt'), dtype=int)
    N1d = np.loadtxt (os.path.join (indir, protein + '_D_selection.txt'), dtype=int)
    N0s   = np.loadtxt ( N0s_f, dtype=int )
    N1s   = np.loadtxt ( N1s_f, dtype=int )
else :
    N0dtmp = np.loadtxt (os.path.join (indir, protein + '_D_neutral.txt'))
    N1dtmp = np.loadtxt (os.path.join (indir, protein + '_D_selection.txt'))
    N0dtmp[np.isnan (N0dtmp)] = -1
    N1dtmp[np.isnan (N1dtmp)] = -1
    N0d = N0dtmp.astype (int)
    N1d = N1dtmp.astype (int)
    
    # singles
    N0stmp = np.loadtxt ( N0s_f )
    N1stmp = np.loadtxt ( N1s_f )
    N0stmp[np.isnan (N0stmp)] = -1
    N1stmp[np.isnan (N1stmp)] = -1
    N0s = N0stmp.astype (int)
    N1s = N1stmp.astype (int)
    
    del N0dtmp
    del N1dtmp
    del N0stmp
    del N1stmp



####
nAA = 20
nreps = 1
L, L = N0d.shape
npos = int (L / nAA)

print (N0d.shape)

# AA to numeric dictionary
aadict = make_aa_dict (stop='*')
aas    = list (aadict.keys ())
print (aas)


# header
outf.write (('\t').join (['aa_seq', 'Nham_aa', 'WT', 'input1','output1A']))
outf.write ('\n')

outf.write (('\t').join ( [all_wt, '0', 'TRUE'] ))
outf.write ( '\t' )
for i in range (nreps) :
    outf.write ( str (input_wt[i]) + '\t' )
    outf.write ( str (output_wt[i]) )
    if i != (nreps - 1) :
        outf.write ( '\t')
outf.write ('\n')

# header
outnt.write (('\t').join (['nt_seq', 'input1','output1A']))
outnt.write ('\n')
outnt.write (convert_aa_to_nt (all_wt))
outnt.write ('\t')
for i in range (nreps) :
    outnt.write ( str (input_wt[i]) + '\t' )
    outnt.write ( str (output_wt[i]) )
    if i != (nreps - 1) :
        outnt.write ( '\t')
outnt.write ('\n')



# now output data to file
for i in range (L) :
    pos_i = int (np.floor (i / nAA))
    for j in range (i+1,L) :
        pos_j = int (np.floor (j / nAA))
        if N0d[i,j] != -1 :
            seq_ij = list (all_wt)
            seq_ij[pos_i] = aas[i % nAA]
            seq_ij[pos_j] = aas[j % nAA]
            outf.write (('').join (seq_ij) + '\t')
            outf.write ( str (2) + '\t' + '\t' )

            # write to nt file
            outnt.write (convert_aa_to_nt ( ('').join (seq_ij)) + '\t')

            outf.write ( str (N0d[i,j] + eta) + '\t' )
            outf.write ( str (N1d[i,j] + eta) )
            outnt.write ( str (N0d[i,j] + eta) + '\t' )
            outnt.write ( str (N1d[i,j] + eta) )

            outf.write ('\n')
            outnt.write ('\n')


# now output data to file
for i in range (L) :
    pos_i = int (np.floor (i / nAA))
    if N0s[i] != -1 :
        seq_i = list (all_wt)
        seq_i[pos_i] = aas[i % nAA]
        outf.write (('').join (seq_i) + '\t')
        outf.write ( str (1) + '\t' + '\t' )
        
        # nt
        outnt.write (convert_aa_to_nt (('').join (seq_i)) + '\t')

        outf.write ( str (int (N0s[i] + eta)) + '\t' )
        outf.write ( str (int (N1s[i] + eta)) )
        outnt.write ( str (int (N0s[i] + eta)) + '\t' )
        outnt.write ( str (int (N1s[i] + eta)) )
       
        outf.write ('\n')
        outnt.write ('\n')
        

