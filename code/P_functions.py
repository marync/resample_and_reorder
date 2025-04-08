import numpy as np
import numba
from numba import njit

def compute_p_values (D_obs, D_sim, rank_singles) :

    # number of loci
    L     = D_obs.shape[0]
    nsims = D_sim.shape[0]

    # P-value matrices
    P     = np.zeros_like (D_obs) * np.nan
    Pasym = np.zeros_like (P) * np.nan
    Psym  = np.zeros_like (P) * np.nan

    for i in range (L) :
            
        if ~np.isnan (rank_singles[i]) :
            rank_1 = int (rank_singles[i])
    
            for j in range (i+1,L) :
                if ~np.isnan (rank_singles[j]) :
                    rank_2 = int (rank_singles[j])
    
                    if ~np.isnan (D_obs[i,j]) :          
                        
                        # sum
                        Dstat = np.abs (D_obs[i,j] + D_obs[j,i])
                        cdf   = np.abs (D_sim[:,rank_2, rank_1] + D_sim[:,rank_1, rank_2])
                        P[i,j] = P[j,i] = np.sum ( cdf >= Dstat )
    
                        # i,j 
                        Dstat      = np.abs (D_obs[i,j])
                        Pasym[i,j] = np.sum (cdf >= Dstat)
                        
                        # j,i
                        Dstat = np.abs (D_obs[j,i])
                        Pasym[j,i] = np.sum (cdf >= Dstat)
                       
                        # conservative test 
                        Psym[i,j] = Psym[j,i] = np.max ([Pasym[i,j], Pasym[j,i]])

    return ((P+1)/(nsims+1)), ((Pasym+1)/(nsims+1)), ((Psym+1)/(nsims+1))


def compute_p_values_two_sided (D_obs, D_sim, rank_singles) :

    # number of loci
    L, L = D_obs.shape
    nsims   = D_sim.shape[0]

    # P-value matrices
    P = np.zeros_like (D_obs) * np.nan
    for i in range (L) :
            
        if ~np.isnan (rank_singles[i]) :
            rank_i = int (rank_singles[i])
    
            for j in range (i+1,L) :
                if ~np.isnan (rank_singles[j]) :
                    rank_j = int (rank_singles[j])
    
                    if ~np.isnan (D_obs[i,j]) :
                        # sum
                        Dstat  = D_obs[i,j] + D_obs[j,i] 
                        cdf    = D_sim[:,rank_i, rank_j] + D_sim[:,rank_j, rank_i]
                        nvals  = np.sum (~np.isnan(cdf))
                        Phigh  = (np.nansum ( cdf >= Dstat ) + 1) / (nvals + 1)
                        Plow   = (np.nansum ( cdf <= Dstat ) + 1) / (nvals + 1)
                        # p-value
                        #P[i,j] = P[j,i] = (np.min ( [Plow, Phigh] ) + 1) / (nvals + 1)
                        P[i,j] = P[j,i] = np.min ([Plow, Phigh, 1])

    return 2*P


@njit
def compute_p_values_integrate (D_obs, D_sim, D_sim_u, row_rank) :

    # number of loci
    L, L = D_obs.shape
    nsims   = D_sim.shape[0]

    # P-value matrices
    P = np.zeros_like (D_obs) * np.nan
    for i in range (L) :
            
        if ~np.isnan (row_rank[i]) :
            rank_i = int (row_rank[i])
    
            for j in range (i+1,L) :
                if ~np.isnan (row_rank[j]) :
                    rank_j = int (row_rank[j])
    
                    if ~np.isnan (D_obs[i,j]) :
                        # sum
                        Dstat  = np.abs ( D_sim_u[:,i,j] + D_sim_u[:,j,i] )
                        cdf    = np.abs ( D_sim[:,rank_i, rank_j] + D_sim[:,rank_j, rank_i] )
                        pvals  = np.zeros (nsims)
                        for k in range (nsims) :
                            pvals[k] = np.sum (cdf >= Dstat[k]) + 1

                        P[i,j] = P[j,i] = np.nanmean (pvals) / (nsims + 1) 
    
    return P


def compute_p_values_trans (D_obs, D_sim, row_rank, col_rank) :

    # number of loci
    Llig, L = D_obs.shape
    nsims   = D_sim.shape[0]

    print (D_sim.shape)

    # P-value matrices
    P    = np.zeros_like (D_obs) * np.nan
    for i in range (Llig) :
            
        if ~np.isnan (row_rank[i]) :
            rank_i = int (row_rank[i])
    
            for j in range (L) :
                if ~np.isnan (col_rank[j]) :
                    rank_j = int (col_rank[j])
    
                    if ~np.isnan (D_obs[i,j]) :
                        # sum
                        Dstat  = np.abs (D_obs[i,j]) 
                        cdf    = np.abs (D_sim[:,rank_i, rank_j])
                        P[i,j] = np.sum ( cdf >= Dstat )
    
    return ((P+1)/(nsims+1))


def compute_p_values_trans_symmetric (D_obs, D_obs_t, D_sim, D_sim_t, row_rank, col_rank) :

    # number of loci
    Llig, L = D_obs.shape
    nsims   = D_sim.shape[0]

    # P-value matrices
    P = np.zeros_like (D_obs) * np.nan
    for i in range (Llig) :
            
        if ~np.isnan (row_rank[i]) :
            rank_i = int (row_rank[i])
    
            for j in range (L) :
                if ~np.isnan (col_rank[j]) :
                    rank_j = int (col_rank[j])
    
                    if ~np.isnan (D_obs[i,j]) :
                        # sum
                        Dstat  = np.abs ( D_obs[i,j] + D_obs_t[j,i] ) 
                        cdf    = np.abs ( D_sim[:,rank_i, rank_j] + D_sim_t[:,rank_j, rank_i] )
                        P[i,j] = np.sum ( cdf >= Dstat )
    
    return ((P+1)/(nsims+1))


#@njit
def compute_p_values_trans_two_sided (D_obs, D_obs_t, D_sim, D_sim_t, row_rank, col_rank) :

    # number of loci
    Llig, L = D_obs.shape
    nsims   = D_sim.shape[0]

    # P-value matrices
    P = np.zeros_like (D_obs) * np.nan
    for i in range (Llig) :
            
        if ~np.isnan (row_rank[i]) :
            rank_i = int (row_rank[i])
    
            for j in range (L) :
                if ~np.isnan (col_rank[j]) :
                    rank_j = int (col_rank[j])
    
                    if ~np.isnan (D_obs[i,j]) :
                        # sum
                        Dstat  = D_obs[i,j] + D_obs_t[j,i] 
                        cdf    = D_sim[:,rank_i, rank_j] + D_sim_t[:,rank_j, rank_i]
                        Phigh  = np.nansum ( cdf >= Dstat )
                        Plow   = np.nansum ( cdf <= Dstat )
                        nvals  = np.sum (~np.isnan(cdf))
                        P[i,j] = (np.min ( [Plow, Phigh] ) + 1) / (nvals + 1)

    outP = 2 * P
    outP[P > 1] = 1.

    return outP
    #return ((P+1)/(nsims+1))

@njit 
def compute_p_values_trans_integrate (D_obs, D_obs_t, D_sim, D_sim_t, D_sim_u, D_sim_ut, row_rank, col_rank) :

    # number of loci
    Llig, L = D_obs.shape
    nsims   = D_sim.shape[0]

    # P-value matrices
    P = np.zeros_like (D_obs) * np.nan
    for i in range (Llig) :
            
        if ~np.isnan (row_rank[i]) :
            rank_i = int (row_rank[i])
    
            for j in range (L) :
                if ~np.isnan (col_rank[j]) :
                    rank_j = int (col_rank[j])
    
                    if ~np.isnan (D_obs[i,j]) :
                        # sum
                        Dstat  = np.abs ( D_sim_u[:,i,j] + D_sim_ut[:,j,i] )
                        cdf    = np.abs ( D_sim[:,rank_i, rank_j] + D_sim_t[:,rank_j, rank_i] )
                        pvals  = np.zeros (nsims)
                        for k in range (nsims) :
                            pvals[k] = np.sum (cdf >= Dstat[k])

                        P[i,j] = np.nanmean (pvals) 
    
    return ((P+1)/(nsims+1))
