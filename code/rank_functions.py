import numpy as np
import random
import copy as cp
import scipy
from numba import njit
import math


##
def transform_ranks (ranks, M) :
    """
    Transform 0-ref ranks to new scale M.
    """
    M_i = np.nanmax (ranks)
    if M_i == 0 or np.isnan (M_i) :
        print (ranks)
    return ranks * ( M / M_i)

##
def compute_transformed_Rmat (Yd, M=None) :

    # rank matrix
    R = np.zeros_like (Yd) * np.nan

    # only not wt indices
    not_wt = np.where (np.sum (~np.isnan (Yd), axis=1) > 0)[0]

    # scale for ranks if not provided
    if M is None :
        M = np.max (np.sum (~np.isnan (Yd), axis=1)) - 1

    for i in not_wt :
        ranks_i = compute_mean_ranks (Yd[i,:])
        R[i,:]  = transform_ranks (ranks_i, M) # transform ranks to same scale 

    return R

##
def compute_mean_ranks (values) :

    # find unique values
    y = cp.deepcopy (values[~np.isnan (values)])
    uvals = np.unique (np.sort (y))

    if len (uvals) < len (y) :
        ranks = np.zeros_like (values) * np.nan

        ct = 0
        for i in uvals :
            idxs = np.where (values == i)[0]
            nidxs = len (idxs)
            if nidxs > 1 :
                ranks[idxs] = ct + (nidxs - 1) / 2
            else :
                ranks[idxs] = ct
            ct += nidxs

    else :
        ranks = compute_ranks_unique (values)

    return ranks

##
def compute_ranks_unique (values) :
    """
    Returns rank matrix when there are no ties.
    """

    ordered = np.argsort (values)
    ranks   = np.zeros_like (values) * np.nan

    for i in range (len (values)) :
        if ~np.isnan (values[ordered[i]]) :
            ranks[ordered[i]] = i

    return ranks

##
def compute_rank_matrix_random ( E, rng, M=None ) :
    """
    If sample=True then sample tied values from uniform
    """

    # number of positions
    nrow, ncol = E.shape

    Rank = np.zeros_like (E) * np.nan
    for i in range (nrow) :
        if np.sum (np.isnan (E[i,:])) != ncol :
            ranks_i = compute_rank_sample (cp.deepcopy (E[i,:]), rng)
            if M is None :
                Rank[i,:] = cp.deepcopy (ranks_i)
            else :
                Rank[i,:] = transform_ranks (ranks_i, M)

    return Rank

##
def compute_rank_sample (values, rng) :
    """
    Resolve ties by sampling uniformly over integers.
    """

    n     = len (values)
    uvals = np.sort (np.unique (values[~np.isnan (values)])) # unique values
    # find ranks
    ranks = np.zeros (n) * np.nan
    count = 0
    for u in uvals :
        idxs  = np.where (values == u)[0]
        nidxs = len (idxs)
        # if more than one value, sample from uniform
        if nidxs > 1 :
            ranks[idxs] = rng.choice (np.arange (0,nidxs,1),
                                      size=nidxs, replace=False)

            # update ranks of current idxs
            ranks[idxs] += count
        else :
            ranks[idxs] = count

        # update rank counter
        count += nidxs

    return ranks


##
def compute_correlation_nan ( y, z ) :

    keep = np.logical_and (~np.isnan (y), ~np.isnan (z))

    return np.corrcoef ( y[keep], z[keep] )[0,1]


##
def compute_rho_mat (R, thres) :
    """
    """

    L = R.shape[0]

    keep = np.where (np.sum (np.isnan (R), axis=1) != L)[0]

    Rho = np.zeros_like (R) * np.nan
    for i in keep :
        for j in keep : 
            if i < j :
                if np.sum (np.logical_and ( ~np.isnan (R[i,:]), ~np.isnan (R[j,:]))) > thres :
                    Rho[i,j] = Rho[j,i] = compute_correlation_nan (R[i,:], R[j,:]) 

    return Rho

##
def compute_rho_mat_trans (R, thres) :
    """
    """

    L, M = R.shape

    keep = np.where (np.sum (np.isnan (R), axis=1) != L)[0]

    Rho = np.zeros ((L,L)) * np.nan
    for i in keep :
        for j in keep : 
            if i < j :
                if np.sum (np.logical_and ( ~np.isnan (R[i,:]), ~np.isnan (R[j,:]))) > thres :
                    Rho[i,j] = Rho[j,i] = compute_correlation_nan (R[i,:], R[j,:]) 

    return Rho


