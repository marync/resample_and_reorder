import numpy as np
import scipy.stats
import pandas

from format_helper_functions import make_aa_dict, find_mutation

def process_mochi (mochi_df, nmuts, nAA=20) :
 
    all_wt = mochi_df['aa_seq'][mochi_df['WT'] == True].to_list ()[0]
    aadict = make_aa_dict ()

    Yd_mochi   = np.zeros ( (nmuts, nmuts) ) * np.nan
    Ystd_mochi = np.zeros_like (Yd_mochi) * np.nan
    Yest_mochi = np.zeros_like (Yd_mochi) * np.nan

    Ys_mochi     = np.zeros ( nmuts ) * np.nan
    Ys_std_mochi = np.zeros_like ( Ys_mochi ) * np.nan
    count = 0
    for index, row in mochi_df.iterrows () :
        muts = find_mutation (row['aa_seq'], all_wt)

        if len (muts) == 2 :
            if row['Nham_aa'] != 2 :
                print ('COWS')
                print (index)

            pp_i, pm_i = muts[0]
            cp_i, cm_i = muts[1]

            Yd_mochi[cp_i*nAA + aadict[cm_i], pp_i*nAA + aadict[pm_i]]   = row['fitness']
            Ystd_mochi[cp_i*nAA + aadict[cm_i], pp_i*nAA + aadict[pm_i]] = row['sigma']
            Yest_mochi[cp_i*nAA + aadict[cm_i], pp_i*nAA + aadict[pm_i]] = row['mean']
            Yd_mochi[pp_i*nAA + aadict[pm_i], cp_i*nAA + aadict[cm_i]]   = row['fitness']
            Ystd_mochi[pp_i*nAA + aadict[pm_i], cp_i*nAA + aadict[cm_i]] = row['sigma']
            Yest_mochi[pp_i*nAA + aadict[pm_i], cp_i*nAA + aadict[cm_i]] = row['mean']

        if len (muts) == 1 :
            pp_i, pm_i = muts[0]        

            Ys_mochi[pp_i*nAA + aadict[pm_i]]     = row['fitness']
            Ys_std_mochi[pp_i*nAA + aadict[pm_i]] = row['sigma']
            
    # compute residuals
    Residuals = (Yd_mochi - Yest_mochi) / Ystd_mochi

    # print R2
    SSres = (Yd_mochi - Yest_mochi)**2
    SStot = (Yd_mochi - np.nanmean (Yd_mochi))**2
    R2    = 1. - np.nansum (SSres)/ np.nansum (SStot)
    print (R2)

    # compute p-values
    pvalues = 2*scipy.stats.norm.cdf (-np.abs (Residuals))

    return pvalues, Residuals, Yd_mochi, Yest_mochi


def process_mochi_trans (data, wt_seq, nlig, nprot, nAA=21) :

    Llig = int (nlig * nAA)
    L    = int (nprot * nAA)

    aadict = make_aa_dict (stop='*') # make aa dictionary

    # organize data from Mochi
    Yd   = np.zeros ( (Llig, L) ) * np.nan
    Ystd = np.zeros_like (Yd) * np.nan
    Yest = np.zeros_like (Yd) * np.nan
    
    # singles
    Ys = np.zeros ( L ) * np.nan
    Yl = np.zeros ( Llig ) * np.nan
    Ys_std = np.zeros_like (Ys) * np.nan
    Yl_std = np.zeros_like (Yl) * np.nan
    
    for index, row in data.iterrows () :

        muts = find_mutation (row['aa_seq'], wt_seq)

        if len (muts) == 2 :
            if row['Nham_aa'] != 2 :
                print ('Wrong number of mutations.')
                print (index)

            pp_i, pm_i = muts[0]
            cp_i, cm_i = muts[1]

            cp_i -= nprot # get index in Jun

            Yd[pp_i*nAA + aadict[pm_i], cp_i*nAA + aadict[cm_i], i]   = row['fitness']
            Ystd[pp_i*nAA + aadict[pm_i], cp_i*nAA + aadict[cm_i], i] = row['sigma']
            Yest[pp_i*nAA + aadict[pm_i], cp_i*nAA + aadict[cm_i], i] = row['mean']

        if len (muts) == 1 :
            if row['Nham_aa'] != 1 :
                print ('Wrong number of mutations.')
                print (index)
                
            pp_i, pm_i = muts[0]  

            if pp_i > (npos-1) :
                pp_i -= npos
                Yl[pp_i*nAA + aadict[pm_i], i]     = row['fitness']
                Yl_std[pp_i*nAA + aadict[pm_i], i] = row['sigma']

            else :
                Ys[pp_i*nAA  + aadict[pm_i], i]    = row['fitness']
                Ys_std[pp_i*nAA + aadict[pm_i], i] = row['sigma']
                Ys_est[pp_i*nAA + aadict[pm_i], i] = row['mean']


    return Yd, Ystd, Yest, Ys, Yl, Ys_std, Yl_std



def find_mochi_alpha (pval_r, pval_m, imat, alpha_sig=.1, alphas=np.logspace (-10,0,1000)) :
    
    # number of non-missing, nulls
    n_nulls = np.sum (np.logical_and (~np.isnan (pval_r), imat == 0))

    # find false positive rates at many alphas
    fp_mochi = np.zeros (len (alphas))
    for i in range (len (alphas)) :
        fp_mochi[i] = np.sum (np.logical_and (pval_m < alphas[i], imat == 0) / n_nulls)
        
    # fp of r and r
    fp_r = np.sum (np.logical_and (pval_r <= alpha_sig, imat == 0)) / n_nulls
    
    # find closest alpha for mochi
    mochi_thres = alphas[np.argmin (np.abs (fp_mochi - fp_r))]
    
    return mochi_thres


def find_mochi_alpha_empirical (pval_r, pval_m, imat, alpha_sig=.1, alphas=np.logspace (-10,0,1000), dthres=5) :
    
    # number of non-missing, nulls
    n_nulls = np.sum (np.logical_and (~np.isnan (pval_r), imat > dthres))

    # find false positive rates at many alphas
    fp_mochi = np.zeros (len (alphas))
    for i in range (len (alphas)) :
        fp_mochi[i] = np.sum (np.logical_and (pval_m <= alphas[i], imat > dthres) / n_nulls)
        
    # fp of r and r
    fp_r = np.sum (np.logical_and (pval_r <= alpha_sig, imat > dthres)) / n_nulls
    
    # find closest alpha for mochi
    mochi_thres = alphas[np.argmin (np.abs (fp_mochi - fp_r))]
    
    return mochi_thres
