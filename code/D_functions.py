import numpy as np
import copy as cp

from rank_functions import *


##
def compute_single_rank_mat (singles, rng=None, nAA=20, M=None) :
    """
    Computes matrix of single effects excluding sites from the same position.
    """

    L    = len (singles) # number of sites
    npos = int (L / nAA)
    wts  = np.where (np.isnan (singles))[0]

    RS = np.zeros ( (L,L) ) * np.nan
    for i in range (npos) :
        # remove same sites
        singles_pos = cp.deepcopy (singles)
        singles_pos[(i*nAA):(i*nAA + nAA)] = np.nan

        # get ranks
        if rng is not None :
            single_ranks = np.reshape ( np.repeat (
                            compute_rank_sample (singles_pos, rng), nAA), (nAA, L), order='F')
        else :
            single_ranks = np.reshape ( np.repeat (
                            compute_mean_ranks (singles_pos), nAA), (nAA, L), order='F')

        if M is None :
            # copy into matrix
            RS[(i*nAA):(i*nAA + nAA), :] = cp.deepcopy ( single_ranks )
        else :
            RS[(i*nAA):(i*nAA + nAA), :] = transform_ranks ( single_ranks, M )

    # set wild type rows to missing
    RS[wts,:] = np.nan

    return RS


def compute_single_rank_mat_fosjun (singles, rng=None, nAA=20, M=None) :
    """ 
    Computes matrix of single effects excluding sites from the same position.
    """

    L    = len (singles) # number of sites
    npos = int (L / nAA)
    print (npos)
 
    prot_length = int (L / 2)
    aa_length   = int (prot_length / nAA)
    wts  = np.where (np.isnan (singles))[0]
    
    RS = np.zeros ( (L,L) ) * np.nan
    for i in range (npos) :
        # remove same sites
        singles_pos = cp.deepcopy (singles)
        
        if i < aa_length :
            singles_pos[:(aa_length*nAA)] = np.nan
        else :
            singles_pos[(aa_length*nAA):] = np.nan
            
        # get ranks
        if rng is not None :
            single_ranks = np.reshape ( np.repeat (
                            compute_rank_sample (singles_pos, rng), nAA), (nAA, L), order='F')
        else :
            single_ranks = np.reshape ( np.repeat (
                            compute_mean_ranks (singles_pos), nAA), (nAA, L), order='F')

        if M is None :
            # copy into matrix
            RS[(i*nAA):(i*nAA + nAA), :] = cp.deepcopy ( single_ranks )
        else :
            RS[(i*nAA):(i*nAA + nAA), :] = transform_ranks ( single_ranks, M ) 

    RS[wts,:] = np.nan
            
    return RS





##
def compute_D_from_ranks (singles, Rmat, ligand=None, protein=None, rng=None, sort=False, M=None, nAA=20) :
    """
    Returns D matrix sorted by ranks.
    """

    if protein is None or protein == 'GB1' :
        Smat = compute_single_rank_mat (singles, rng, M=M, nAA=nAA) 
    elif protein == 'fosjun' :
        Smat = compute_single_rank_mat_fosjun (singles, rng, M=M, nAA=nAA)
    elif protein == 'trans' :
        L_lig, L = Rmat.shape
        if rng is not None :
            sranks = transform_ranks (compute_rank_sample ( singles, rng=rng ), M=M)
        else :
            sranks = transform_ranks (compute_mean_ranks ( singles ), M=M)
    
        Smat = np.reshape (np.repeat (sranks, L_lig), (L_lig, L), order='F')

    # compute Dmat
    Dmat = Rmat - Smat

    if sort :
        if ligand is None :
            sorted_Dmat = cp.deepcopy ( Dmat[np.argsort (singles), :][:, np.argsort (singles)] )
        else :
            sorted_Dmat = cp.deepcopy ( Dmat[np.argsort (ligand), :][:, np.argsort (singles)] ) 
    
        return sorted_Dmat
    else :
        return Dmat



##
def compute_single_rbar_mat (ranks, nAA=20, M=None) :
    """
    Computes matrix of single effects excluding sites from the same position.
    """

    L    = len (ranks) # number of sites
    npos = int (L / nAA)
    wts  = np.where (np.isnan (ranks))[0]

    RS = np.zeros ( (L,L) ) * np.nan
    for i in range (npos) :
        # remove same sites
        ranks_pos = cp.deepcopy (ranks)
        ranks_pos[(i*nAA):(i*nAA + nAA)] = np.nan

        reranked = np.reshape ( np.repeat (
                               compute_ranks_unique (ranks), nAA), (nAA, L), order='F')

        if M is None :
            # copy into matrix
            RS[(i*nAA):(i*nAA + nAA), :] = cp.deepcopy ( reranked )
        else :
            RS[(i*nAA):(i*nAA + nAA), :] = transform_ranks ( reranked, M )
    
    RS[wts,:] = np.nan

    return RS



##
def compute_Dbar_from_ranks (Rmat, sort=False, M=None, nAA=20) :
    """
    """

    rbars = np.nanmean (Rmat, axis=0)
    ranks = compute_ranks_unique (rbars)
    Dmat  = Rmat - compute_single_rbar_mat (ranks, M=M)

    if sort :
        sorted_Dmat = cp.deepcopy ( Dmat[np.argsort (rbars), :][:, np.argsort (rbars)] )
        return sorted_Dmat
    else :
        return Dmat




##
def impute_matrix_nn (A, rng, k=2) :
    """
    Imputes matrix A with a randomly sampled value among the at-most k-nearest
    neighbors.
    """

    # number loci
    L1, L2 = A.shape

    # imputed matrix
    Aimp = cp.deepcopy (A)

    # dictionary of indices
    Idic = generate_nn_indices (k)

    # iterate through rows
    for i in range (L1) :
        idxs_na = np.where (np.isnan (A[i,:]))[0]
        for j in idxs_na :
            val = np.nan
            kj  = 1
            while np.isnan (val) and kj <= k :
                xis = i + Idic[kj][:,0]
                yis = j + Idic[kj][:,1]

                # remove values which are out of range
                xis[np.logical_or (xis < 0, yis < 0)] = -1
                xis[np.logical_or (xis >= L1, yis >= L2)] = -1
                xs = xis[xis != -1]
                ys = yis[xis != -1]

                # set of potential values
                pool = A[xs, ys]
                if np.sum (~np.isnan (pool)) > 0 :
                    val = rng.choice ( pool[~np.isnan (pool)] )

                kj += 1

            Aimp[i,j] = val

    return Aimp

##
def generate_nn_indices (k) :
    """
    Generates a dictionary of indices for imputation.
    """

    # build index arrays
    I_dict   = dict ()
    old_list = list ()
    for i in range (1,k+1) :
        cur_list = list ()
        for j in range (-i,i+1) :
            for k in range (-i,i+1) :
                if [j,k] not in old_list :
                    cur_list.append ([j,k])

        I_dict[i] = np.array (cur_list)
        old_list  = cp.deepcopy (cur_list)

    return I_dict


