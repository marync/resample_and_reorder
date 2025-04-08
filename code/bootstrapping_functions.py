import copy as cp
import numpy as np

##
def set_to_zero (A, val=0, NA=None) :
    """
    Sets missing values (NA) to val. 
    """

    A_tmp = cp.deepcopy (A)
    if NA is None :
        A_tmp[np.isnan (A)] = val
    else :
        A_tmp[A == NA] = val

    return A_tmp


##
def bootstrap_fitness_singles (reads_neu_sing, reads_sel_sing,
                               nsims, rng,
                               N0=True,
                               eta=0, NA=None, inside=True, threshold=0) :

    L = len (reads_neu_sing) # number of site x AA
    neu_int = set_to_zero (reads_neu_sing, NA=NA)
    sel_int = set_to_zero (reads_sel_sing, NA=NA)

    # no pseudo-count
    if inside :
        N_1_sing = rng.poisson ( sel_int + eta, size=(nsims, L) )
        if N0 :
            N_0_sing = rng.poisson ( neu_int + eta, size=(nsims, L) ) # sample N0
        else :
            N_0_sing = cp.deepcopy ( neu_int )                        # don't sample N0
    
    # pseduo-count
    else :
        N_1_sing = rng.poisson ( sel_int, size=(nsims, L) ) + eta
        if N0 :
            N_0_sing = rng.poisson ( neu_int, size=(nsims, L) ) + eta
        else :
            N_0_sing = neu_int + eta

    # filter on threshold
    if threshold > 0 :
        if inside :
            N_0_sing[N_0_sing < threshold] = 0
        else :
            N_0_sing[N_0_sing < (threshold-eta)] = 0
    
    # calculate fitness
    Singles_sim = np.transpose (N_1_sing / N_0_sing)
    
    # remove values for which N0 = 0
    Singles_sim[np.isinf (Singles_sim)] = np.nan

    # remove wt counts
    if NA is None :
        Singles_sim[np.isnan (reads_neu_sing),:] = np.nan
    else :
        Singles_sim[reads_neu_sing == -1,:] = np.nan


    return Singles_sim


##
def bootstrap_fitness_doubles (reads_neu_doub, reads_sel_doub, nsims, rng,
                               N0=False, eta=0, NA=None, inside=True, threshold=0) :
    """
    Samples the read counts based on a poisson read model.
    """

    L = reads_neu_doub.shape[0] # number of site x AA

    # get rid of NA's
    Doubles_sel_int = cp.deepcopy (reads_sel_doub)
    Doubles_neu_int = cp.deepcopy (reads_neu_doub)
    if NA is None :
        Doubles_sel_int[np.isnan (reads_sel_doub)] = 0
        Doubles_neu_int[np.isnan (reads_neu_doub)] = 0
    else :
        Doubles_sel_int[reads_sel_doub == NA] = 0
        Doubles_neu_int[reads_neu_doub == NA] = 0

    # simulate each b
    Doubles_sim = np.ones ( (L, L, nsims), dtype=float ) * np.nan
    for i in range (nsims) :
        # sample N_1 and N_0
        if inside :
            N_1_doub = rng.poisson ( np.triu (Doubles_sel_int) + eta )   # sample upper triangle
            if N0 :
                N_0_doub = rng.poisson ( np.triu (Doubles_neu_int) + eta )
            else :
                N_0_doub = cp.deepcopy (reads_neu_doub)
        else :
            N_1_doub = rng.poisson ( np.triu (Doubles_sel_int) )     # sample upper triangle
            N_0_doub = rng.poisson ( np.triu (Doubles_neu_int) )

        # compute Y and make symmetric
        N_1_doub[np.tril_indices (L, k=-1)] = 0
        N_0_doub[np.tril_indices (L, k=-1)] = 0
        N_0_doub[Doubles_neu_int == 0] = 0
        N_1_doub[Doubles_neu_int == 0] = 0 

        # make symmetric
        N_1_doub += np.transpose (N_1_doub)
        N_0_doub += np.transpose (N_0_doub)

        # compute Y
        Y_doub = (N_1_doub + eta) / (N_0_doub + eta)

        # remove inf values
        if NA is None :
            Y_doub[np.isnan (reads_neu_doub)] = np.nan
        else :
            Y_doub[reads_neu_doub == NA] = np.nan

        if eta == 0 :
            Y_doub[np.isinf (Y_doub)] = np.nan

        # set reads to missing that are below threshod
        if threshold > 0 :
            Y_doub[N_0_doub < threshold] = np.nan
        else :
            Y_doub[N_0_doub <= 0] = np.nan

        # save
        Doubles_sim[:,:,i] = cp.deepcopy (Y_doub)

    return Doubles_sim


##
def bootstrap_fitness_doubles_trans (reads_neu_doub, reads_sel_doub,
                                     nsims, rng, eta=0, NA=None, inside=True, threshold=0) :
    """
    Samples the read counts based on a poisson read model for an asymmetric (trans) DMS.
    """

    L_lig, L = reads_neu_doub.shape  # number of site x AA

    # set missing values to 0
    Doubles_sel = cp.deepcopy (reads_sel_doub)
    Doubles_neu = cp.deepcopy (reads_neu_doub)

    if NA is None :
        Doubles_sel[np.isnan (reads_sel_doub)] = 0
        Doubles_neu[np.isnan (reads_neu_doub)] = 0
    else :
        Doubles_sel[reads_sel_doub == NA] = 0
        Doubles_neu[reads_neu_doub == NA] = 0

    # simulate each b
    Doubles_sim = np.ones ( (L_lig, L, nsims), dtype=float ) * np.nan
    for i in range (nsims) :
        # sample N_1 and N_0
        if inside :
            N_1_doub = rng.poisson ( Doubles_sel + eta ) # sample upper triangle
            N_0_doub = rng.poisson ( Doubles_neu + eta )
        else :
            N_1_doub = rng.poisson ( Doubles_sel ) # sample upper triangle
            N_0_doub = rng.poisson ( Doubles_neu )

        # compute Y and make symmetric
        Y_doub = (N_1_doub + eta) / (N_0_doub + eta)

        # remove values where initially 0
        Y_doub[Doubles_neu == 0] = np.nan

        # set reads to missing that are below threshod
        if threshold > 0 :
            Y_doub[N_0_doub < threshold] = np.nan
        else :
            Y_doub[N_0_doub <= 0] = np.nan

        # save
        Doubles_sim[:,:,i] = cp.deepcopy (Y_doub)

    return Doubles_sim


##
def bootstrap_fitness_normal (means, stds, nsims, rng) :
    """
    Bootstrap fitness values based on a normal error model.
    """

    if means.ndim == 1 :
        L = len (means)
        Ysims = rng.normal (loc=means, scale=stds, size=(nsims, L))

    else :
        Llig, L = means.shape
        Ysims   = rng.normal (loc=means, scale=stds, size=(nsims, Llig, L))

    return Ysims

