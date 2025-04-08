import numpy as np
import os
import copy as cp
import pandas
import scipy


# from https://www.petercollingridge.co.uk/tutorials/bioinformatics/codon-table/
def make_codon_table (codon_to_aa=True) :

    bases = 'tcag'
    codons = [a + b + c for a in bases for b in bases for c in bases]
    amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
    if codon_to_aa :
        codon_table = dict(zip(codons, amino_acids))
    else :
        codon_table = dict(zip(amino_acids, codons))

    return codon_table


def find_mutation (query, ref) :
    L    = len (query)
    muts = [(x, query[x]) for x in range (L) if query[x] != ref[x]]
    
    return muts


def make_aa_dict (code='one', stop=None,  gap=None, aa_to_num=True) :
    if code == 'three' :
        aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'ILE', 'LEU', 'LYS',
               'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIS']
    elif code == 'one' :
        aas = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'H']
    else :
        return 'Unknown AA dictionary.'

    if gap is not None :
        aas.append (gap)

    if stop is not None :
        aas.append (stop)

    if aa_to_num :
        aadict = dict (zip (aas, np.arange (0, len (aas),1)))
    else :
        aadict = dict (zip (np.arange (0, len (aas),1), aas))

    return aadict


def compute_effect_mat_from_df (df, npos, offset=0, nAA=20) :
    """
    Convert data frame into a 4-d matrix of double mutant phenotypes.
    """

    # first position, second position, first AA, second AA
    effectmat    = np.zeros ( (npos, npos, nAA, nAA) )*np.nan 
    nrows, ncols = df.shape

    for index, row in df.iterrows () :
        pos_i = row['Mut1Position'] - offset
        pos_j = row['Mut2Position'] - offset
        aa_i  = row['aa1_num']
        aa_j  = row['aa2_num']
        effectmat[pos_i, pos_j, aa_i, aa_j] = row['SelectionCount'] / row['InputCount']
        effectmat[pos_j, pos_i, aa_j, aa_i] = row['SelectionCount'] / row['InputCount']

    return effectmat


def compute_read_mats_from_df (df, npos, offset=0, nAA=20) :
    """
    Convert data frame into a 4-d matrix of double mutant phenotypes.
    """

    # first position, second position, first AA, second AA
    reads_sel    = np.zeros ( (npos, npos, nAA, nAA) )*np.nan
    reads_neu    = np.zeros ( (npos, npos, nAA, nAA) )*np.nan
    nrows, ncols = df.shape

    for index, row in df.iterrows () :
        pos_i = row['Mut1Position'] - offset
        pos_j = row['Mut2Position'] - offset
        aa_i  = row['aa1_num']
        aa_j  = row['aa2_num']
        reads_sel[pos_i, pos_j, aa_i, aa_j] = row['SelectionCount']
        reads_sel[pos_j, pos_i, aa_j, aa_i] = row['SelectionCount']
        reads_neu[pos_i, pos_j, aa_i, aa_j] = row['InputCount']
        reads_neu[pos_j, pos_i, aa_j, aa_i] = row['InputCount']

    return reads_sel, reads_neu


def flatten_effect_mat (E, npos, nAA=20) :
    """
    Make a 2-d matrix from a 4-d matrix.
    """

    bigE = np.zeros ((npos*nAA, npos*nAA)) * np.nan
    for i in range (npos) :
        for j in range (npos) :
            bigE[(i*nAA):(i*nAA + nAA),(j*nAA):(j*nAA + nAA)] = cp.deepcopy (E[i,j,:,:])

    return bigE



def format_energies_mat_from_df (df, npos, variables, nAA=20, offset=0) :

    # format single effects
    dim = len (variables)
    singles    = np.zeros ( (npos*nAA, dim) )
    singlesMat = np.zeros ( (npos, nAA, dim))

    ymat = df[variables].to_numpy ()

    count = 0
    for i in range (npos) :
        for j in range (nAA) :
            idx = np.where ( np.logical_and (df['pos'] == i+offset, df['aa_num'] == j) )[0]
            if len (idx) == 1 :
                value = cp.deepcopy (ymat[idx[0],:])
            else :
                value = np.zeros (dim) * np.nan

            # assign value
            singles[count,:] = singlesMat[i,j,:] = value

            # reset
            value = np.zeros (dim) * np.nan
            count += 1

    return singles, singlesMat


def format_singles_mat_from_df (df, npos, nAA=20, offset=0) :

    # format single effects
    singles    = np.zeros (npos*nAA)
    singlesMat = np.zeros ((npos, nAA))

    count = 0
    for i in range (npos) :
        for j in range (nAA) :
            idx = np.where ( np.logical_and (df['Position'] == i+offset, df['aa_num'] == j) )[0]
            if len (idx) == 1 :
                value = df['SelectionCount'][idx[0]] / df['InputCount'][idx[0]]
            else :
                value = np.nan

            # assign value
            singles[count]  = singlesMat[i,j] = value

            # reset
            value = np.nan
            count += 1

    return singles, singlesMat


def format_singles_reads_from_df (df, npos, nAA=20, offset=0) :

    # format single effects
    reads_sel = np.zeros (npos*nAA) * np.nan
    reads_neu = np.zeros (npos*nAA) * np.nan

    count = 0
    for i in range (npos) :
        for j in range (nAA) :
            idx = np.where ( np.logical_and (df['Position'] == i+offset, df['aa_num'] == j) )[0]
            if len (idx) == 1 :
                reads_sel[count] = df['SelectionCount'][idx[0]]
                reads_neu[count] = df['InputCount'][idx[0]]

            count += 1

    return reads_sel, reads_neu


#def format_double_reads_fos_jun (df, npos, nAA=20, offset=0, posno=1) :
#
#    for i, line in enumerate (df):
#        #if i == 0:
#        #    header = line.strip().split(',')
#        #    continue
#
#        row = line.strip().split(',')
#        id1 = row[header.index("id1")]
#        id2 = row[header.index("id2")]
#        pos1 = str(int(row[header.index("pos1")])-1)
#        pos2 = str(int(row[header.index("pos2")])+31)
#        aa1 = row[header.index("mut1")]
#        aa2 = row[header.index("mut2")]
#        sum_in  = str(int(row[header.index("d_i1")]) + int(row[header.index("d_i2")]) + int(row[header.index("d_i3")]))
#        sum_out = str(int(row[header.index("d_o1")]) + int(row[header.index("d_o2")]) + int(row[header.index("d_o3")]))
#        doub_row = [pos1, pos2, aa1, aa2, sum_in, sum_out, "\n"]
#        doub_out.write("\t".join(doub_row))


def compute_read_mats_fos_jun (df, npos, offset=0, nAA=20) :
    """
    Convert data frame into a 4-d matrix of double mutant phenotypes.
    """

    # first position, second position, first AA, second AA
    reads_sel    = np.zeros ( (npos, npos, nAA, nAA) )*np.nan
    reads_neu    = np.zeros ( (npos, npos, nAA, nAA) )*np.nan
    nrows, ncols = df.shape

    pre_cols  = ['d_i' + str (i) for i in range (1,4)]
    post_cols = ['d_o' + str (i) for i in range (1,4)]

    aadict = make_aa_dict ()

    df['sum_pre']  = df[pre_cols[0]] + df[pre_cols[1]] + df[pre_cols[2]]
    df['sum_post'] = df[post_cols[0]] + df[post_cols[1]] + df[post_cols[2]]

    df['aa1_num'] = df['mut1'].map (aadict)
    df['aa2_num'] = df['mut2'].map (aadict)

    for index, row in df.iterrows () :
        pos_i = row['pos1'] - offset
        pos_j = row['pos2'] - offset
        aa_i  = row['aa1_num']
        aa_j  = row['aa2_num']
        reads_sel[pos_i, pos_j, aa_i, aa_j] = row['sum_post']
        reads_sel[pos_j, pos_i, aa_j, aa_i] = row['sum_post']
        reads_neu[pos_i, pos_j, aa_i, aa_j] = row['sum_pre']
        reads_neu[pos_j, pos_i, aa_j, aa_i] = row['sum_pre']

    return reads_sel, reads_neu




def format_singles_reads_fos_jun (df, npos, nAA=20, offset=0, posno=1) :
    """

    """

    # format single effects
    reads_sel = np.zeros (npos*nAA) * np.nan
    reads_neu = np.zeros (npos*nAA) * np.nan

    pos_col = 'pos' + str (posno)
    aa_col  = 'mut' + str (posno)
    pre_cols  = ['s' + str (posno) + '_i' + str (i) for i in range (1,4)]
    post_cols = ['s' + str (posno) + '_o' + str (i) for i in range (1,4)]


    df['sum_pre']  = df[pre_cols[0]] + df[pre_cols[1]] + df[pre_cols[2]]
    df['sum_post'] = df[post_cols[0]] + df[post_cols[1]] + df[post_cols[2]]

    aadict = make_aa_dict ()

    df['aa_num'] = df[aa_col].map (aadict)

    count = 0
    for i in range (npos) :
        for j in range (nAA) :

            idx = np.where ( np.logical_and (df[pos_col] == i+offset, df['aa_num'] == j) )[0]
            if len (idx) > 1 :
                reads_sel[count] = df['sum_post'][idx[0]]
                reads_neu[count] = df['sum_pre'][idx[0]]

            count += 1

    return reads_sel, reads_neu



def compute_effect_mat_olson (mat, npos, nAA=20) :
    """
    """
    
    # first position, second position, first AA, second AA
    effectmat = np.zeros ( (npos, npos, nAA, nAA) )*np.nan 
    nrows     = mat.shape[0]
    
    for i in range (nrows) :
        row_i = np.array (mat[i,:], dtype=int)
        effectmat[row_i[0] - 2, row_i[1] - 2, row_i[2], row_i[3]] = (mat[i,5] / mat[i,4])
        effectmat[row_i[1] - 2, row_i[0] - 2, row_i[3], row_i[2]] = (mat[i,5] / mat[i,4])

    return effectmat


def compute_rho_mat (effectmat) :
    """
    Should eventually deal with ties.
    """
    
    Rho = np.zeros ( (nAA, nAA) )
    for i in range (nAA) :
        for j in range (nAA) :
            Rho[i,j] = scipy.stats.spearmanr (np.ndarray.flatten (effectmat[i,:,:]),
                                                                  np.ndarray.flatten (effectmat[j,:,:]),
                                                                  nan_policy='omit').statistic

    return Rho


def compute_additive_matrix (singles, npos, nAA=20) :
    """
    S_ij = S_i + S_j
    """

    Smat = np.zeros ( (npos, npos, nAA, nAA) )*np.nan
    for i in range (npos) :
        for j in range (npos) :
            if j != i :
                for k in range (nAA) :
                    for l in range (nAA) :
                        Smat[i,j,k,l] = singles[i*nAA + k] + singles[j*nAA + l]

    #Smat[Smat == 0] = np.nan

    return Smat


def flatten_additive_matrix (S, npos, nAA=20) :

    # make a big matrix
    bigS = np.zeros ((npos*nAA, npos*nAA))*np.nan
    for i in range (npos) :
        for j in range (npos) :
            if i <= j :
                bigS[(i*nAA):(i*nAA + nAA),(j*nAA):(j*nAA + nAA)] = cp.deepcopy (Smat[i,j,:,:])
                bigS[(j*nAA):(j*nAA + nAA),(i*nAA):(i*nAA + nAA)] = cp.deepcopy (np.transpose (Smat[i,j,:,:]))

    return bigS




