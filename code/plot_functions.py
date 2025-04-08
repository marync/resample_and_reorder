import os
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import seaborn
import scipy.stats
from numba import njit


def compute_accuracy_recall ( P, Dist, distance_thres, quantile=True, alphas=np.linspace (0,1,100) ) :
    
    Acc  = np.zeros ( len (alphas) )
    Prop = np.zeros ( len (alphas) )
    
    if quantile == True :
        vals = np.ndarray.flatten (cp.deepcopy (P))
        p_thresholds = np.quantile (vals[~np.isnan (vals)], alphas)
    
    else :
        p_thresholds = cp.deepcopy (alphas)
            
    for i in range ( len (p_thresholds) ) :
        nhits   = np.sum ( np.logical_and (P <= p_thresholds[i], Dist <= distance_thres))
        Acc[i]  = nhits / np.sum (P <= p_thresholds[i])
        Prop[i] = nhits / np.sum (np.logical_and (Dist <= distance_thres, ~np.isnan (P)))

    return Acc, Prop


def compute_auc ( P, Dist, distance_thres, quantile=True, alphas=np.linspace (0,1,100) ) :
    
    Tps = np.zeros ( len (alphas) )
    Fps = np.zeros ( len (alphas) )
    
    if quantile == True :
        vals = np.ndarray.flatten (cp.deepcopy (P))
        p_thresholds = np.quantile (vals[~np.isnan (vals)], alphas)
        print (np.nanmin (p_thresholds))
    
    else :
        p_thresholds = cp.deepcopy (alphas)
            
    for i in range ( len (p_thresholds) ) :
        nhits  = np.sum ( np.logical_and (P <= p_thresholds[i], Dist <= distance_thres))
        Tps[i] = nhits / np.sum (np.logical_and (~np.isnan (P), Dist <= distance_thres))
        
        nfalse = np.sum ( np.logical_and (P <= p_thresholds[i], Dist > distance_thres))
        Fps[i] = nfalse / np.sum (np.logical_and (~np.isnan (P), Dist > distance_thres))

    return Tps, Fps

def compute_enrichment_accuracy (P, Dist, distance_thres, quantile=True, alphas=np.linspace (0,1,100), alpha_bh=.05 ) :
    Acc  = np.zeros ( len (alphas) )
    Prop = np.zeros ( len (alphas) )
    
    # p-value thresholds
    if quantile == True :
        vals = np.ndarray.flatten (cp.deepcopy (P))
        p_thresholds = np.quantile (vals[~np.isnan (vals)], alphas)    
    else :
        p_thresholds = cp.deepcopy (alphas)        
    
    # calculate accuracy for each threshold
    for i in range ( len (p_thresholds) ) :
        Ep_i, Et_i = compute_enrichment (P, alpha=p_thresholds[i]) # enrichment
        
        fdr_i  = scipy.stats.false_discovery_control (np.ndarray.flatten (Ep_i), method='bh') # fdr
        if np.sum (fdr_i < .1) > 0 :
            fdr_thres_i = np.nanmax (np.ndarray.flatten (Ep_i)[fdr_i < alpha_bh])
        
            nhits = np.sum (np.logical_and (Ep_i <= fdr_thres_i, Dist <= distance_thres))
            Acc[i]  = nhits / np.sum (Ep_i <= fdr_thres_i)
            Prop[i] = nhits / np.sum (np.logical_and (~np.isnan (Ep_i), Dist <= distance_thres))
        else :
            Acc[i] = np.nan
            Prop[i] = np.nan

    return Acc, Prop



def log_bins (vals, nbins=20) :
    
    bins = np.logspace (np.log10 (np.nanmin (vals)), np.log10 (np.nanmax (vals)), nbins)
    
    return bins

def plot_enrichment_fos (Emat, Dmat, Dist, Ps=None, Mmat=None,
                         ligand='jun', protein='fos', start=1,
                         outlabel='', save=False, outdir='.', figsize=(3,3)) :
    
    # plot dimensions
    plt.rcParams["figure.figsize"] = (2,2)
    
    # axes
    npos, npos = Emat.shape
    xpos = np.arange (start, start + npos, 1)
    
    # compute average sign matrix
    avgSign = compute_avg_sign_matrix (Dmat, Ps)
    # multiple test correction
    if Mmat is None :
        Mmat = multiple_test_correct (Emat)
        
    # plot avg sign to get colorbar
    fig, ax = plt.subplots ()
    hm = ax.imshow (avgSign, cmap='Spectral', vmin=-1, vmax=1)
    plt.show ()

    # spectral color map
    spec = mpl.colormaps['Spectral']
    norm = plt.Normalize(-1, 1)

    # find contacts and nearby things
    C5 = np.zeros_like (Dist)
    C8 = np.zeros_like (Dist)

    # Actual plot
    plt.rcParams["figure.figsize"] = figsize
    fig, ax = plt.subplots ()

    # distances for plotting
    C5[Dist <= 5] = 1
    C8[np.logical_and (Dist > 5, Dist <= 8)] = 1
    
    # ghost plot to create template
    Cnon = cp.deepcopy (C5)
    Cnon[~np.isnan (C5)] = np.nan
    ax.imshow (Cnon)

    # within 8
    x, y = np.where ( C8 == 1 )
    for i in range (len (x)) :
        ax.add_patch( Rectangle((y[i]-.5, x[i]-.5), 1, 1, fill=True,
                                color='lightgray', alpha=0.4, lw=.5))# label=r'$\leq 5\AA$')    
    # within 5
    x, y = np.where ( C5 == 1 )
    for i in range (len (x)) :
        ax.add_patch( Rectangle((y[i]-.5, x[i]-.5), 1, 1, fill=True,
                                color='lightgray', lw=.5))# label=r'$\leq 5\AA$')

    # significant
    x, y = np.where (Mmat <= .01)
    ax.scatter (y,x, marker='o', edgecolor='black', linewidth=.2, color=spec( norm (avgSign[x,y])),
                     s=(-np.log10 (Emat[x,y])))

    # "less" significant
    x, y = np.where (np.logical_and (Mmat > .01, Mmat <= .1))
    ax.scatter (y,x, marker='o', color=spec( norm (avgSign[x,y])), linewidth=0,
                     s=(-np.log10 (Emat[x,y])))

    # colorbar
    cbar = plt.colorbar (hm, ax=ax, location='top', shrink=0.5, pad=0.02)
    cbar.ax.tick_params (labelsize=6, length=3, pad=.5)
    cbar.set_label (label=r'avg. sign $\hat{D}_{ij}$', labelpad=5, size=6) 

    # Create the figure
    ticklocs   = np.arange (1, npos, 4)
    plt.xticks (ticklocs, xpos[ticklocs])
    plt.yticks (ticklocs, xpos[ticklocs])
    plt.xlabel (ligand.upper () + r' position $i$', labelpad=1 )
    plt.ylabel (protein.upper () + r' position $i$', labelpad=1 )
    ax.tick_params(axis='both', which='major', length=2, pad=1)


    if save :
        plt.savefig (os.path.join (outdir, outlabel + 'dij_improved.pdf'), bbox_inches='tight')
        plt.close ()
    else :
        plt.show ()


def plot_enrichment_fos_svg (Emat, Dmat, Dist, Ps=None, Mmat=None,
                             ligand='jun', protein='fos', start=1,
                             outlabel='', save=False, outdir='.', alpha=.1,
                             figsize=(2,2)) :
    
    # plot dimensions
    plt.rcParams["figure.figsize"] = (2,2)
    
    # axes
    npos, npos = Emat.shape
    xpos = np.arange (start, start + npos, 1)
    
    # compute average sign matrix
    avgSign = compute_avg_sign_matrix (Dmat, Ps=Ps, alpha=alpha)
    # multiple test correction
    if Mmat is None :
        Mmat = multiple_test_correct (Emat)
        
    # plot avg sign to get colorbar
    fig, ax = plt.subplots ()
    hm = ax.imshow (avgSign, cmap='Spectral', vmin=-1, vmax=1)
    plt.show ()

    # spectral color map
    spec = mpl.colormaps['Spectral']
    norm = plt.Normalize(-1, 1)

    # find contacts and nearby things
    C5 = np.zeros_like (Dist)
    C8 = np.zeros_like (Dist)

    # Actual plot
    plt.rcParams["figure.figsize"] = figsize
    fig, ax = plt.subplots ()

    # distances for plotting
    C5[Dist <= 5] = 1
    C8[np.logical_and (Dist > 5, Dist <= 8)] = 1
    
    # ghost plot to create template
    Cnon = cp.deepcopy (C5)
    Cnon[~np.isnan (C5)] = np.nan
    ax.imshow (Cnon)

    # within 8
    x, y = np.where ( C8 == 1 )
    for i in range (len (x)) :
        ax.add_patch( Rectangle((y[i]-.5, x[i]-.5), 1, 1, fill=True,
                                color='lightgray', alpha=0.4, lw=.5)) 
    # within 5
    x, y = np.where ( C5 == 1 )
    for i in range (len (x)) :
        ax.add_patch( Rectangle((y[i]-.5, x[i]-.5), 1, 1, fill=True,
                                color='lightgray', lw=.5)) 

    # significant
    x, y = np.where (Mmat <= .01)
    for i in range (len (x)) :
        x_i = x[i]
        y_i = y[i]
        if np.isclose (Emat[x_i, y_i], 0) :
            Emat[x_i, y_i] = np.min ([1e-10, np.nanmin (Emat[Emat > 0])])
        
        plt.scatter (y_i,x_i, marker='o', edgecolor='black', linewidth=.2,
                     color=spec( norm (avgSign[x_i,y_i])),
                     s=(-np.log10 (Emat[x_i,y_i]))**1.1)

    print ()
    # "less" significant
    x, y = np.where (np.logical_and (Mmat > .01, Mmat <= .1))
    for i in range (len (x)) :
        x_i = x[i]
        y_i = y[i]
        
        if np.isclose (Emat[x_i, y_i], 0) :
            Emat[x_i, y_i] = np.min ([1e-10, np.nanmin (Emat[Emat > 0])])
        
        plt.scatter (y_i, x_i, marker='o', linewidth=0, 
                     color=spec( norm (avgSign[x_i,y_i])),
                     s=(-np.log10 (Emat[x_i,y_i]))**1.1)

        #print ((x_i, y_i, Emat[x_i,y_i]))
    #ax.scatter (y,x, marker='o', color=spec( norm (avgSign[x,y])),
    #            s=(-np.log10 (Emat[x,y])*1.5))

    # colorbar
    cbar = plt.colorbar (hm, ax=ax, location='right', shrink=0.5, pad=0.05)
    cbar.ax.tick_params (labelsize=6, length=2, pad=1)
    cbar.set_label (label=r'avg. sign $\hat{D}$', labelpad=2, size=6) 

    ax.locator_params(axis='x', nbins=5)
    ax.tick_params (length=2)
    ax.xaxis.set_tick_params (pad=2)
    ax.yaxis.set_tick_params (pad=1.5)
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.tick_params(which='minor', length=1.5)
    
    # Create the figure
    ticklocs   = np.arange (1, npos, 4)
    plt.xticks (ticklocs, xpos[ticklocs], fontsize=6)
    plt.yticks (ticklocs, xpos[ticklocs], fontsize=6)
    plt.xlabel (ligand[0].upper () + ligand[1:].lower () + r' position $i$', labelpad=1 )
    plt.ylabel (protein[0].upper () + protein[1:].lower () + r' position $j$', labelpad=1 )


    if save :
        plt.savefig (os.path.join (outdir, outlabel + 'dij_fosjun.svg'),
                  bbox_inches='tight', format='svg', transparent=True, dpi=600)
        plt.close ()
    else :
        plt.show ()


def make_fpr_figure_individual (P, Imat, Dmat, ys, ls,
                                outlabel='', alpha=.1, ithres=5,
                                ybinsize=50, lbinsize=50, save=False, outdir='.',
                                figsize=(.75,.75),
                                ylabel=r'$\hat Y_i^{jun}$', llabel=r'$\hat Y_j^{fos}$',
                                conmax=None, fpmax=None, powermax=None,
                                simulation=False, colorbar=True, labelcolor=True) :

    fs = 7
    boxthres = 10
    Pvals   = cp.deepcopy (P)
    bigImat = cp.deepcopy (Imat)

    print (Pvals.shape)

    ylabel += ' bin'
    llabel += ' bin' 
 
    # create sign matrix
    SignD   = cp.deepcopy (Dmat)
    SignD[Dmat < 0] = -1
    SignD[Dmat > 0] = 1
    SignD[Pvals > alpha] = np.nan
    
    Hits    = np.zeros_like (Pvals)
    FPs     = np.zeros_like (Pvals)
    Tot_fp  = np.zeros_like (Pvals)
    Tot     = np.zeros_like (Pvals)

    # sort matrices by single effects
    Pval_sort    = cp.deepcopy (Pvals[np.argsort (ls),:][:,np.argsort (ys)])
    bigImat_sort = cp.deepcopy (bigImat[np.argsort (ls),:][:,np.argsort (ys)])
    Sign_sort    = cp.deepcopy (SignD[np.argsort (ls),:][:,np.argsort (ys)])
    
    if simulation :
        # true positives
        Hits[np.logical_and (Pval_sort < alpha, np.abs (bigImat_sort) > ithres)] = 1
        # total number contacts
        Tot[np.logical_and (~np.isnan (Pval_sort), np.abs (bigImat_sort) > ithres)] = 1
        # false positives
        FPs[np.logical_and (Pval_sort < alpha, np.abs (bigImat_sort) <= ithres)] = 1
        # total number of negatives
        Tot_fp[np.logical_and (~np.isnan (Pval_sort), np.abs (bigImat_sort) <= ithres)] = 1
    else :
        Hits[np.logical_and (Pval_sort < alpha, bigImat_sort <= ithres)] = 1
        Tot[np.logical_and (~np.isnan (Pval_sort), bigImat_sort <= ithres)] = 1
        FPs[np.logical_and (Pval_sort < alpha, bigImat_sort > ithres)] = 1
        Tot_fp[np.logical_and (~np.isnan (Pval_sort), bigImat_sort > ithres)] = 1

    print (np.sum (np.isnan (Pval_sort)))


    # set missing values to missing
    Hits[np.isnan (Pval_sort)] = np.nan
    Tot[np.isnan (Pval_sort)] = np.nan
    FPs[np.isnan (Pval_sort)] = np.nan
    Tot_fp[np.isnan (Pval_sort)] = np.nan
    
    print (np.sum (np.isnan (Hits)))
    print (np.sum (np.isnan (Tot)))

    # now compute power and false positives
    nys   = np.sum (~np.isnan (ys))
    ybins = np.linspace (0, nys, int (nys / ybinsize))
    nls = np.sum (~np.isnan (ls))
    lbins = np.linspace (0, nls, int (nls / lbinsize))

    Power  = np.zeros ( (len (lbins) - 1, len (ybins) - 1) )
    Posit  = np.zeros_like (Power)
    Totavg = np.zeros_like (Posit)
    Totsum = np.zeros_like (Power)
    Savg   = np.zeros_like (Posit)
    Nhits  = np.zeros_like (Power)
    Nwrong = np.zeros_like (Power)
    for i in range ( len (lbins) - 1 ) :
        for j in range ( len (ybins) - 1 ) :
            
            yl = int (np.floor (ybins[j]))
            yh = int (np.ceil (ybins[j+1]))
                
            ll = int (np.floor (lbins[i]))
            lh = int (np.ceil (lbins[i+1]))
 
            num_ij = np.nansum (Hits[ll:lh,:][:,yl:yh])
            den_ij = np.nansum (Tot[ll:lh,:][:,yl:yh])
            Power[i,j] = num_ij / den_ij
            Nhits[i,j] = num_ij
            
            num_ij = np.nansum (FPs[ll:lh,:][:,yl:yh])
            den_ij = np.nansum (Tot_fp[ll:lh,:][:,yl:yh])
            Posit[i,j] = num_ij / den_ij
            Nwrong[i,j] = num_ij

            Savg[i,j] = np.nanmean ( Sign_sort[ll:lh,:][:,yl:yh] )
            Totij = cp.deepcopy (Tot[ll:lh,:][:,yl:yh])
            Pij   = cp.deepcopy (Pval_sort[ll:lh,:][:,yl:yh])
            Totavg[i,j] = np.nansum ( Totij )
            Totsum[i,j] = np.nansum (Tot[ll:lh,:][:,yl:yh]) # Totavg[i,j] 
            Totavg[i,j] /= np.sum (~np.isnan (Pij))

    # make figure
    plt.rcParams["figure.figsize"] = figsize

    fig, axs = plt.subplots ()
    print (conmax) 
    h0 = axs.imshow (Totavg, cmap='gray_r')# vmin=0, vmax=conmax)# , vmax=.45)

    print (np.nanmax (Totavg))

    # label axes
    axs.set_ylabel (llabel, labelpad=1, fontsize=fs)
    axs.set_xlabel (ylabel, labelpad=1, fontsize=fs)
    axs.tick_params (length=2)
    axs.yaxis.set_tick_params (pad=2)
    axs.xaxis.set_tick_params (pad=2)
    axs.set_xticks ( np.arange (0, len (ybins)-1, 2),
                     np.array (np.arange (0, len (ybins)-1, 2), dtype=int), fontsize=6)
    axs.set_yticks ( np.arange (0, len (lbins)-1, 2),
                     np.array (np.arange (0, len (lbins)-1, 2), dtype=int), fontsize=6)
   
    if colorbar : 
        cbar = fig.colorbar (h0, ax=axs, shrink=0.75)#, label='Prop.') #label=r'$|\lambda_{ij}| > 0$')
        cbar.ax.tick_params (labelsize=6, length=3, pad=.5)
        if labelcolor :
            cbar.set_label (r'prop. $|\lambda_{ij}| > 0$', size=fs, labelpad=1)
    
    xs, ys = np.where (Totsum < boxthres)
    for j in range (2) : 
        for k in range (len (xs)) :
            axs.add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                           edgecolor='black', linestyle='dotted', alpha=1., lw=.2))
    
    if save :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_prop_contacts.svg'),
                     bbox_inches='tight', format='svg', transparent=True, dpi=600)
        plt.close ()
    else :
        plt.show ()

    
    # POWER
    fig, axs = plt.subplots ()
    print ('power:')
    print (np.nanmax (Power)) 
    h1 = axs.imshow (Power, cmap='rocket_r', vmin=0, vmax=powermax)
    # label axes
    axs.set_ylabel (llabel, labelpad=1, fontsize=fs)
    axs.set_xlabel (ylabel, labelpad=1, fontsize=fs)
    axs.tick_params (length=2)
    axs.yaxis.set_tick_params (pad=2)
    axs.xaxis.set_tick_params (pad=2)
    axs.set_xticks ( np.arange (0, len (ybins)-1, 2),
                     np.array (np.arange (0, len (ybins)-1, 2), dtype=int), fontsize=6)
    axs.set_yticks ( np.arange (0, len (lbins)-1, 2),
                     np.array (np.arange (0, len (lbins)-1, 2), dtype=int), fontsize=6)
    
    #fig.colorbar (h1, ax=axs[0,1], shrink=0.75)#, label='t.p.r.')
    xs, ys = np.where (Totsum < boxthres)
    for j in range (2) : 
        for k in range (len (xs)) :
            axs.add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                           edgecolor='black', linestyle='dotted', alpha=1., lw=.2))
    
    if colorbar : 
        cbar = fig.colorbar (h1, ax=axs, shrink=0.75)#, label='Prop.') #label=r'$|\lambda_{ij}| > 0$')
        cbar.ax.tick_params (labelsize=6, length=3, pad=.5)
        if labelcolor :
            cbar.set_label (r'true pos. rate', size=fs, labelpad=1)
        #cbar.ax.set_xticklabels( np.round (np.arange (0,4,1)*.05,2))    
 
    if save :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_power.svg'),
                     bbox_inches='tight', format='svg', transparent=True, dpi=600)
        plt.close ()
    else :
        plt.show ()
    
    # FALSE POSITIVE RATE
    fig, axs = plt.subplots ()
    
    h3 = axs.imshow (Posit, cmap='Blues', vmin=0, vmax=fpmax)#, vmax=.18)#, vmax=.241) #norm=mpl.colors.LogNorm ())
    print (np.nanmax (Posit))
    # label axes
    axs.set_ylabel (llabel, labelpad=1, fontsize=fs)
    axs.set_xlabel (ylabel, labelpad=1, fontsize=fs)
    axs.tick_params (length=2)
    axs.yaxis.set_tick_params (pad=2)
    axs.xaxis.set_tick_params (pad=2)
    axs.set_xticks ( np.arange (0, len (ybins)-1, 2),
                     np.array (np.arange (0, len (ybins)-1, 2), dtype=int), fontsize=6)
    axs.set_yticks ( np.arange (0, len (lbins)-1, 2),
                     np.array (np.arange (0, len (lbins)-1, 2), dtype=int), fontsize=6)
    
    if colorbar : 
        cbar = fig.colorbar (h3, ax=axs, shrink=0.75)#, label='Prop.') #label=r'$|\lambda_{ij}| > 0$')
        cbar.ax.tick_params (labelsize=6, length=3, pad=.5)
        if labelcolor :
            cbar.set_label (r'false pos. rate', size=fs, labelpad=1)
    
    if save :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_false_positive_rate.svg'),
                     bbox_inches='tight', format='svg', transparent=True, dpi=600)
        plt.close ()
    else :
        plt.show ()


    fig, axs = plt.subplots ()
    
    h4 = axs.imshow (Savg, cmap='Spectral', vmin=-1, vmax=1)
    #fig.colorbar (h11, ax=axs, shrink=0.75)#, label=r'Avg. $sign (\hat D_{ij})$')
    axs.set_ylabel (llabel, labelpad=1, fontsize=fs)
    axs.set_xlabel (ylabel, labelpad=1, fontsize=fs)
    axs.tick_params (length=2)
    axs.yaxis.set_tick_params (pad=2)
    axs.xaxis.set_tick_params (pad=2)
    axs.set_xticks ( np.arange (0, len (ybins)-1, 2),
                     np.array (np.arange (0, len (ybins)-1, 2), dtype=int), fontsize=6)
    axs.set_yticks ( np.arange (0, len (lbins)-1, 2),
                     np.array (np.arange (0, len (lbins)-1, 2), dtype=int), fontsize=6)
    
    xs, ys = np.where (Nhits < boxthres)
    for k in range (len (xs)) :
        axs.add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                       edgecolor='black', linestyle='dotted', alpha=1., lw=.2))

    if colorbar : 
        cbar = fig.colorbar (h4, ax=axs, shrink=0.75)#, label='Prop.') #label=r'$|\lambda_{ij}| > 0$')
        cbar.ax.tick_params (labelsize=6, length=3, pad=.5)
        if labelcolor :
            cbar.set_label (r'avg. sign $\hat D$', size=fs, labelpad=1)
    
    if save :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_sign.svg'),
                     bbox_inches='tight', format='svg', transparent=True, dpi=600)
        plt.close ()
    else :
        plt.show ()


def make_fpr_figure_four (P, Imat, Dmat, ys, ls,
                                outlabel='', alpha=.1, ithres=5,
                                ybinsize=50, lbinsize=50, save=False, outdir='.',
                                figsize=(.75,.75),
                                ylabel=r'$\hat Y_i^{jun}$', llabel=r'$\hat Y_i^{fos}$',
                                conmax=None, fpmax=None, powermax=None,
                                simulation=False, colorbar=True, labelcolor=True) :

    boxthres = 10
    Pvals   = cp.deepcopy (P)
    bigImat = cp.deepcopy (Imat)

    print (Pvals.shape)

    ylabel += ' bin'
    llabel += ' bin' 
 
    # create sign matrix
    SignD   = cp.deepcopy (Dmat)
    SignD[Dmat < 0] = -1
    SignD[Dmat > 0] = 1
    SignD[Pvals > alpha] = np.nan
    
    Hits    = np.zeros_like (Pvals)
    FPs     = np.zeros_like (Pvals)
    Tot_fp  = np.zeros_like (Pvals)
    Tot     = np.zeros_like (Pvals)

    # sort matrices by single effects
    Pval_sort    = cp.deepcopy (Pvals[np.argsort (ls),:][:,np.argsort (ys)])
    bigImat_sort = cp.deepcopy (bigImat[np.argsort (ls),:][:,np.argsort (ys)])
    Sign_sort    = cp.deepcopy (SignD[np.argsort (ls),:][:,np.argsort (ys)])
    
    if simulation :
        # true positives
        Hits[np.logical_and (Pval_sort < alpha, np.abs (bigImat_sort) > ithres)] = 1
        # total number contacts
        Tot[np.logical_and (~np.isnan (Pval_sort), np.abs (bigImat_sort) > ithres)] = 1
        # false positives
        FPs[np.logical_and (Pval_sort < alpha, np.abs (bigImat_sort) <= ithres)] = 1
        # total number of negatives
        Tot_fp[np.logical_and (~np.isnan (Pval_sort), np.abs (bigImat_sort) <= ithres)] = 1
    else :
        Hits[np.logical_and (Pval_sort < alpha, bigImat_sort <= ithres)] = 1
        Tot[np.logical_and (~np.isnan (Pval_sort), bigImat_sort <= ithres)] = 1
        FPs[np.logical_and (Pval_sort < alpha, bigImat_sort > ithres)] = 1
        Tot_fp[np.logical_and (~np.isnan (Pval_sort), bigImat_sort > ithres)] = 1

    # set missing values to missing
    Hits[np.isnan (Pval_sort)] = np.nan
    Tot[np.isnan (Pval_sort)] = np.nan
    FPs[np.isnan (Pval_sort)] = np.nan
    Tot_fp[np.isnan (Pval_sort)] = np.nan

    # now compute power and false positives
    nys   = np.sum (~np.isnan (ys))
    ybins = np.linspace (0, nys, int (nys / ybinsize))
    nls = np.sum (~np.isnan (ls))
    lbins = np.linspace (0, nls, int (nls / lbinsize))

    Power  = np.zeros ( (len (lbins) - 1, len (ybins) - 1) )
    Posit  = np.zeros_like (Power)
    Totavg = np.zeros_like (Posit)
    Totsum = np.zeros_like (Power)
    Savg   = np.zeros_like (Posit)
    Nhits  = np.zeros_like (Power)
    Nwrong = np.zeros_like (Power)
    for i in range ( len (lbins) - 1 ) :
        for j in range ( len (ybins) - 1 ) :
            
            yl = int (np.floor (ybins[j]))
            yh = int (np.ceil (ybins[j+1]))
                
            ll = int (np.floor (lbins[i]))
            lh = int (np.ceil (lbins[i+1]))
 
            num_ij = np.nansum (Hits[ll:lh,:][:,yl:yh])
            den_ij = np.nansum (Tot[ll:lh,:][:,yl:yh])
            Power[i,j] = num_ij / den_ij
            Nhits[i,j] = num_ij
            
            num_ij = np.nansum (FPs[ll:lh,:][:,yl:yh])
            den_ij = np.nansum (Tot_fp[ll:lh,:][:,yl:yh])
            Posit[i,j] = num_ij / den_ij
            Nwrong[i,j] = num_ij

            Savg[i,j] = np.nanmean ( Sign_sort[ll:lh,:][:,yl:yh] )
            Totij = cp.deepcopy (Tot[ll:lh,:][:,yl:yh])
            Pij   = cp.deepcopy (Pval_sort[ll:lh,:][:,yl:yh])
            Totavg[i,j] = np.nansum ( Totij )
            Totsum[i,j] = np.nansum (Tot[ll:lh,:][:,yl:yh]) # Totavg[i,j] 
            Totavg[i,j] /= np.sum (~np.isnan (Pij))

    # make figure
    plt.rcParams["figure.figsize"] = figsize

    fig, axs = plt.subplots (2, 2, sharex=True, sharey=True, constrained_layout=True)
    print (conmax) 
    h0 = axs[0,0].imshow (Totavg, cmap='gray_r')# vmin=0, vmax=conmax)# , vmax=.45)

    print (np.nanmax (Totavg))

    # label axes
    axs[0,0].set_ylabel (llabel, labelpad=0)
    #axs[0,0].set_xlabel (ylabel, labelpad=0)
    axs[0,0].tick_params (length=2)
    axs[0,0].yaxis.set_tick_params (pad=2)
    axs[0,0].xaxis.set_tick_params (pad=2)
    axs[0,0].set_xticks ( np.arange (0, len (ybins)-1, 2),
                     np.array (np.arange (0, len (ybins)-1, 2), dtype=int), fontsize=6)
    axs[0,0].set_yticks ( np.arange (0, len (lbins)-1, 2),
                     np.array (np.arange (0, len (lbins)-1, 2), dtype=int), fontsize=6)
   
    axs[0,0].set_title (r'prop. $|\lambda_{ij}| > 0$', fontsize=7, pad=3)

    if colorbar : 
        cbar = fig.colorbar (h0, ax=axs[0,0], shrink=0.75, aspect=10)
        cbar.ax.tick_params (labelsize=6, length=3, pad=.5)
        #if labelcolor :
        #    cbar.set_label (r'prop. contacts', size=8, labelpad=1)
    
    xs, ys = np.where (Totsum < boxthres)
    for j in range (2) : 
        for k in range (len (xs)) :
            axs[0,0].add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                           edgecolor='black', linestyle='dotted', alpha=1., lw=.2))

    
    # POWER
    print ('power:')
    print (np.nanmax (Power)) 
    h1 = axs[0,1].imshow (Power, cmap='rocket_r', vmin=0, vmax=powermax)
    # label axes
    #axs[0,1].set_ylabel (llabel, labelpad=0)
    #axs[0,1].set_xlabel (ylabel, labelpad=0)
    axs[0,1].tick_params (length=2)
    axs[0,1].yaxis.set_tick_params (pad=2)
    axs[0,1].xaxis.set_tick_params (pad=2)
    axs[0,1].set_xticks ( np.arange (0, len (ybins)-1, 2),
                     np.array (np.arange (0, len (ybins)-1, 2), dtype=int), fontsize=6)
    axs[0,1].set_yticks ( np.arange (0, len (lbins)-1, 2),
                     np.array (np.arange (0, len (lbins)-1, 2), dtype=int), fontsize=6)
    
    xs, ys = np.where (Totsum < boxthres)
    for j in range (2) : 
        for k in range (len (xs)) :
            axs[0,1].add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                           edgecolor='black', linestyle='dotted', alpha=1., lw=.2))
    
    axs[0,1].set_title ('true pos. rate', fontsize=7, pad=3)
    if colorbar : 
        cbar = fig.colorbar (h1, ax=axs[0,1], shrink=0.75, aspect=10)
        cbar.ax.tick_params (labelsize=6, length=3, pad=.5)
        #if labelcolor :
        #    cbar.set_label (r'$tpr$', size=8, labelpad=1)
        #cbar.ax.set_xticklabels( np.round (np.arange (0,4,1)*.05,2))    
 
    
    # FALSE POSITIVE RATE
    h3 = axs[1,0].imshow (Posit, cmap='Blues', vmin=0, vmax=fpmax)#, vmax=.18)#, vmax=.241) #norm=mpl.colors.LogNorm ())
    print (np.nanmax (Posit))
    # label axes
    axs[1,0].set_ylabel (llabel, labelpad=0)
    axs[1,0].set_xlabel (ylabel, labelpad=0)
    axs[1,0].tick_params (length=2)
    axs[1,0].yaxis.set_tick_params (pad=2)
    axs[1,0].xaxis.set_tick_params (pad=2)
    axs[1,0].set_xticks ( np.arange (0, len (ybins)-1, 2),
                     np.array (np.arange (0, len (ybins)-1, 2), dtype=int), fontsize=6)
    axs[1,0].set_yticks ( np.arange (0, len (lbins)-1, 2),
                     np.array (np.arange (0, len (lbins)-1, 2), dtype=int), fontsize=6)
    axs[1,0].set_title ('false pos. rate', fontsize=7, pad=3)
    
    if colorbar : 
        cbar = fig.colorbar (h3, ax=axs[1,0], shrink=0.75, aspect=10)
        cbar.ax.tick_params (labelsize=6, length=3, pad=.5)
        #if labelcolor :
        #    cbar.set_label (r'$fpr$', size=8, labelpad=1)
    
    
    h4 = axs[1,1].imshow (Savg, cmap='Spectral', vmin=-1, vmax=1)
    axs[1,1].set_xlabel (ylabel, labelpad=0)
    axs[1,1].tick_params (length=2)
    axs[1,1].yaxis.set_tick_params (pad=2)
    axs[1,1].xaxis.set_tick_params (pad=2)
    axs[1,1].set_xticks ( np.arange (0, len (ybins)-1, 2),
                     np.array (np.arange (0, len (ybins)-1, 2), dtype=int), fontsize=6)
    axs[1,1].set_yticks ( np.arange (0, len (lbins)-1, 2),
                     np.array (np.arange (0, len (lbins)-1, 2), dtype=int), fontsize=6)
    
    xs, ys = np.where (Nhits < boxthres)
    for k in range (len (xs)) :
        axs[1,1].add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                       edgecolor='black', linestyle='dotted', alpha=1., lw=.2))


    axs[1,1].set_title (r'avg. sign $\hat D$', fontsize=7, pad=3)
    if colorbar : 
        cbar = fig.colorbar (h4, ax=axs[1,1], shrink=0.75, aspect=10)
        cbar.ax.tick_params (labelsize=6, length=3, pad=.5)
        #if labelcolor :
        #    cbar.set_label (r'avg. sign $\hat D$', size=8, labelpad=1)
    
    if save :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_four_plots.pdf'),
                     bbox_inches='tight', format='pdf', transparent=True, dpi=600)
        plt.close ()
    else :
        plt.show ()



def make_fpr_figure_contacts_asym (P, Imat, Dmat, ys, ls,
                                   outlabel='', alpha=.1, ithres=5, fpmax=None, svg=False,
                                   ybinsize=50, lbinsize=50, save=False, outdir='.') :

    Pvals   = cp.deepcopy (P)
    bigImat = cp.deepcopy (Imat)
    
    # create sign matrix
    SignD   = cp.deepcopy (Dmat)
    SignD[Dmat < 0] = -1
    SignD[Dmat > 0] = 1
    SignD[Pvals > alpha] = np.nan
    
    Hits    = np.zeros_like (Pvals)
    FPs     = np.zeros_like (Pvals)
    Tot_fp  = np.zeros_like (Pvals)
    Tot     = np.zeros_like (Pvals)

    # sort matrices by single effects
    Pval_sort    = cp.deepcopy (Pvals[np.argsort (ls),:][:,np.argsort (ys)])
    bigImat_sort = cp.deepcopy (bigImat[np.argsort (ls),:][:,np.argsort (ys)])
    Sign_sort    = cp.deepcopy (SignD[np.argsort (ls),:][:,np.argsort (ys)])
    
    # true positives
    Hits[np.logical_and (Pval_sort < alpha, bigImat_sort <= ithres)] = 1
    Hits[np.isnan (Pval_sort)] = np.nan
    # total number contacts
    Tot[np.logical_and (~np.isnan (Pval_sort), bigImat_sort <= ithres)] = 1
            
    # total false positives
    FPs[np.logical_and (Pval_sort < alpha, bigImat_sort > ithres)] = 1
    FPs[np.isnan (Pval_sort)] = np.nan
    # total zeros
    Tot_fp = np.zeros_like (Pval_sort)
    Tot_fp[np.logical_and (~np.isnan (Pval_sort), bigImat_sort > ithres)] = 1

    # now compute power and false positives
    nys   = np.sum (~np.isnan (ys))
    ybins = np.linspace (0, nys, int (nys / ybinsize))

    nls = np.sum (~np.isnan (ls))
    lbins = np.linspace (0, nls, int (nls / lbinsize))

    Power  = np.zeros ( (len (lbins) - 1, len (ybins) - 1) )
    Posit  = np.zeros_like (Power)
    Totavg = np.zeros_like (Posit)
    Totsum = np.zeros_like (Power)
    Savg   = np.zeros_like (Posit)
    Nhits  = np.zeros_like (Power)
    Nwrong = np.zeros_like (Power)
    for i in range ( len (lbins) - 1 ) :
        for j in range ( len (ybins) - 1 ) :
            
            yl = int (np.floor (ybins[j]))
            yh = int (np.ceil (ybins[j+1]))
                
            ll = int (np.floor (lbins[i]))
            lh = int (np.ceil (lbins[i+1]))
            
            num_ij = np.nansum (Hits[ll:lh,:][:,yl:yh])
            den_ij = np.nansum (Tot[ll:lh,:][:,yl:yh])
            if np.isclose (den_ij, 0) :
                Power[i,j] = np.nan
            else :
                Power[i,j] = num_ij / den_ij
            Nhits[i,j] = num_ij
            
            num_ij = np.nansum (FPs[ll:lh,:][:,yl:yh])
            den_ij = np.nansum (Tot_fp[ll:lh,:][:,yl:yh])
            if np.isclose (den_ij, 0) :
                Posit[i,j] = np.nan
            else :
                Posit[i,j] = num_ij / den_ij
            
            # number of false positives
            Nwrong[i,j] = num_ij

            # average sign
            Savg[i,j] = np.nanmean ( Sign_sort[ll:lh,:][:,yl:yh] )
            
            # total number of true positives
            Totavg[i,j] = np.nansum (Tot[ll:lh,:][:,yl:yh]) / np.sum (~np.isnan (Pval_sort)[ll:lh,:][:,yl:yh])
            Totsum[i,j] = np.nansum (Tot[ll:lh,:][:,yl:yh])
      
    # make figure
    plt.rcParams["figure.figsize"] = (3,2.5)

    fig, axs = plt.subplots (2, 2, sharex=True, sharey=True, constrained_layout=True)
   
    h0 = axs[0,0].imshow (Totavg, cmap='gray_r')
    fig.colorbar (h0, ax=axs[0,0], shrink=0.75)
    
    h1 = axs[0,1].imshow (Power, cmap='rocket_r', vmin=0, vmax=1)
    fig.colorbar (h1, ax=axs[0,1], shrink=0.75)#, label='t.p.r.')

    h2 = axs[1,0].imshow (Posit, cmap='Blues', vmin=0, vmax=fpmax) #norm=mpl.colors.LogNorm ())
    fig.colorbar (h2, ax=axs[1,0], shrink=0.75)#, label='f.p.r.')
    print ('max fp: ' + str (np.nanmax (Posit)))


    h11 = axs[1,1].imshow (Savg, cmap='Spectral', vmin=-1, vmax=1)
    fig.colorbar (h11, ax=axs[1,1], shrink=0.75)#, label=r'Avg. $sign (\hat D_{ij})$')

    for i in range (2) :
        axs[i,0].set_yticks (np.arange (0,len (ybins)-1,2), np.arange (0,len (ybins)-1,2))
        for j in range (2) :
            axs[i,j].set_xticks (np.arange (0,len (lbins)-1,2), np.arange (0,len (ybins)-1,2))
        
    xs, ys = np.where (Totsum < 5)
    for j in range (2) : 
        for k in range (len (xs)) :
            axs[0,j].add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                                edgecolor='black', linestyle='dotted', alpha=1., lw=.5))
    
    xs, ys = np.where (Nhits < 5)
    for k in range (len (xs)) :
        axs[1,1].add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                            edgecolor='black', linestyle='dotted', alpha=1., lw=.5))


    # label axes
    for i in range (2) :
        axs[i,0].set_ylabel (r'$\hat Y_j^{fos}$ rank bin', labelpad=0)
        axs[1,i].set_xlabel (r'$\hat Y_i^{jun}$ rank bin', labelpad=0)

    axs[0,0].set_title (r'prop. $d_{ij} \leq 5\AA$', size=8)
    axs[0,1].set_title (r'pseudo-tpr', size=8)
    axs[1,0].set_title (r'pseudo-fpr', size=8)
    axs[1,1].set_title (r'avg. sign $\hat D_{ij}$', size=8)

    for i in range (2) :
        for j in range (2) :
            axs[i,j].tick_params (length=2)
            axs[i,j].yaxis.set_tick_params (pad=2)
        
    if save and svg :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_sorted_pos_four.svg'),
                     format='svg', dpi=600, bbox_inches='tight')
        plt.close ()
    elif save :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_sorted_pos_four.pdf'), bbox_inches='tight')
        plt.close ()
    else :
        plt.show ()
        
        
    plt.rcParams["figure.figsize"] = (3,2)

    fig, axs = plt.subplots (1,2, sharex=False, sharey=True, constrained_layout=True)
   
    myp = axs[0].imshow (Nhits, cmap='rocket_r')
    myo = axs[1].imshow (Nwrong, cmap='Blues')

    axs[0].set_yticks (np.arange (0,len (ybins)-1,2), np.arange (0,len (ybins)-1,2))
    axs[0].set_ylabel (r'$\hat Y_j^{fos}$ rank bin', labelpad=0)
    for i in range (2) :
        axs[i].set_xticks (np.arange (0,len (lbins)-1,2), np.arange (0,len (ybins)-1,2))
        axs[i].set_xlabel (r'$\hat Y_i^{jun}$ rank bin', labelpad=0)
        fig.colorbar ([myp,myo][i], ax=axs[i], orientation='horizontal', location='top',
                        shrink=0.75, label=['# pseudo-t.p.','# pseudo-f.p.'][i])
    if save :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_numhits.pdf'), bbox_inches='tight')
        plt.close ()
    else :
        plt.show ()


def compute_meta_p (P, omit=True, nthres=None) :
    
    Nobs = np.sum (~np.isnan (P), axis=2)
    LogP = -2.*np.nansum (np.log (P), axis=2)
    
    outP = 1 - scipy.stats.chi2.cdf(LogP, 2*Nobs, loc=0, scale=1)
    if omit :
        if nthres is None :
            outP[Nobs < np.nanmax (Nobs)] = np.nan
        else :
            outP[Nobs < nthres] = np.nan

    return outP

def compute_enrichment (Pmat, alpha, nAA=21, positive=False, Sign=None, symmetric=False) :
    
    # dimensions of P matrix
    Llig, L = Pmat.shape
    nlig = int (Llig / nAA)
    npos = int (L / nAA)
    
    # threshold matrix
    if positive and Sign is not None :
        Pthres = 1. * np.logical_and (Pmat <= alpha, Sign > 0)
    else :
        Pthres = 1. * (Pmat <= alpha)
    
    # set missign values to missing
    Pthres[np.isnan (Pmat)] = np.nan
    
    # prop positives
    prop = np.nansum (Pthres) / np.nansum (~np.isnan (Pmat))

    if symmetric :
        M = np.sum (~np.isnan (Pmat)) / 2
        n = np.nansum (Pthres) / 2
    else :
        M = np.sum (~np.isnan (Pmat))
        n = np.nansum (Pthres)

    # do binomial test
    Pbin = np.zeros ( (nlig, npos) ) * np.nan
    Ptot = np.zeros ((nlig, npos)) *np.nan
    for i in range (nlig) :
        for j in range (npos) :
            sig_ij = np.nansum (Pthres[(i*nAA):(i*nAA + nAA),:][:,j*nAA:(j*nAA + nAA)])
            tot_ij = np.sum (~np.isnan (Pthres[(i*nAA):(i*nAA + nAA),:][:,j*nAA:(j*nAA + nAA)]))
            
            if n == 0 :
                Pbin[i,j] = 1
            else :
                Pbin[i,j] = 1. - scipy.stats.hypergeom.cdf(sig_ij, M, n, tot_ij, loc=0)

            Ptot[i,j] = sig_ij
    
    if symmetric :
        np.fill_diagonal (Pbin, np.nan)
        np.fill_diagonal (Ptot, np.nan)

    return Pbin, Ptot



def plot_enrichment (Emat, Dmat, Dist, Posmat=None, Mmat=None, ligand='jun', protein='fos', Ps=None,
                     start=1, startlig=1, save=False, outdir='.', outlabel='prot', liglabels=None,
                     alpha=.1, pdf=True, aspect=1) :
   
    # plot dimensions
    plt.rcParams["figure.figsize"] = (3.3,3.3)
  
    nlig, npos = Emat.shape
  
    # axes
    xpos = np.arange (start, start + npos, 1)
    ypos = np.arange (startlig, startlig + nlig, 1)

    # compute average sign matrix
    avgSign = compute_avg_sign_matrix (Dmat, Ps=Ps, alpha=alpha)
    # multiple test correction
    if Mmat is None :
        Mmat = multiple_test_correct (Emat)

    if Posmat is not None :
        PM = multiple_test_correct (Posmat)
        
    # plot avg sign to get colorbar
    fig, ax = plt.subplots ()
    hm = ax.imshow (avgSign, cmap='Spectral', vmin=-1, vmax=1, aspect=aspect)
    plt.show ()

    # spectral color map
    spec = mpl.colormaps['Spectral']
    norm = plt.Normalize(-1, 1)

    # find contacts and nearby things
    C5 = np.zeros_like (Dist)
    C8 = np.zeros_like (Dist)

    # Actual plot
    fig, ax = plt.subplots ()

    # distances for plotting
    C5[Dist <= 5] = 1
    C8[np.logical_and (Dist > 5, Dist <= 8)] = 1
    
    # ghost plot to create template
    Cnon = cp.deepcopy (C5)
    Cnon[~np.isnan (C5)] = np.nan
    ax.imshow (Cnon, aspect=aspect)

    # within 8
    x, y = np.where ( C8 == 1 )
    for i in range (len (x)) :
        ax.add_patch( Rectangle((y[i]-.5, x[i]-.5), 1, 1, fill=True,
                                color='lightgray', alpha=0.4, lw=.5))# label=r'$\leq 5\AA$')    
    # within 5
    x, y = np.where ( C5 == 1 )
    for i in range (len (x)) :
        ax.add_patch( Rectangle((y[i]-.5, x[i]-.5), 1, 1, fill=True,
                                color='lightgray', lw=.5))# label=r'$\leq 5\AA$')

    if Posmat is not None :
        x, y = np.where ( PM <= .01 )
        for i in range (len (x)) :
            ax.add_patch( Rectangle((y[i]-.5, x[i]-.5), 1, 1, fill=False,
                                    color='black', lw=.2))# label=r'$\leq 5\AA$')
        x, y = np.where ( np.logical_and (PM > .01, PM <= .1) )
        for i in range (len (x)) :
            ax.add_patch( Rectangle((y[i]-.5, x[i]-.5), 1, 1, fill=False,
                                    color='black', linestyle='dotted', lw=.2))# label=r'$\leq 5\AA$')


    # significant
    x, y = np.where (Mmat <= .01)
    print (x)
    for i in range (len (x)) :
        x_i = x[i]
        y_i = y[i]

        if np.isclose (Emat[x_i, y_i], 0) :
            Emat[x_i, y_i] = np.min ([1e-10, np.nanmin (Emat[Emat > 0])])
        
        ax.scatter (y_i,x_i, marker='o', edgecolor='black', linewidth=.2,
                     color=spec( norm (avgSign[x_i,y_i])),
                     s=(-np.log10 (Emat[x_i,y_i]))**1.1)

    # "less" significant
    x, y = np.where (np.logical_and (Mmat > .01, Mmat <= .1))
    for i in range (len (x)) :
        x_i = x[i]
        y_i = y[i]
        plt.scatter (y_i, x_i, marker='o', edgecolor='white', linewidth=0, 
                     color=spec( norm (avgSign[x_i,y_i])),
                     s=(-np.log10 (Emat[x_i,y_i]))**1.1)


    # colorbar
    #cbar = plt.colorbar (hm, ax=ax, location='top', shrink=0.5, pad=0.02)
    #cbar.ax.tick_params (labelsize=10)
    #cbar.set_label (label=r'Avg. sign $\hat{D}_{ij}$', labelpad=10, size=12) 
    ax.locator_params(axis='x', nbins=5)
    ax.tick_params (length=2)
    ax.xaxis.set_tick_params (pad=2)
    ax.yaxis.set_tick_params (pad=1.5)


    # Create the figure
    ticklocs   = np.arange (0, npos, 4)
    plt.xticks (ticklocs, xpos[ticklocs], fontsize=6)
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.tick_params(which='minor', length=1.5)
    
    if liglabels is not None :
        ticklocs = np.arange (0, nlig, int (nlig / len (liglabels)))
        plt.yticks (ticklocs, liglabels, fontsize=6)
    else :
        ticklocs   = np.arange (0, nlig, 5)
        plt.yticks (ticklocs, ypos[ticklocs])
    plt.xlabel (r'PDZ3 position $i$', labelpad=1, fontsize=8 )
    plt.ylabel (r'CRIPT position $j$', labelpad=1, fontsize=8 )
    #plt.ylabel (protein.upper () + r' position $i$' )
    print ('cows')
    

    if save :
        if pdf :
            plt.savefig (os.path.join (outdir, outlabel + 'dij_enrichment.pdf'),
                  bbox_inches='tight', format='pdf', transparent=True, dpi=600)
        else : 
            plt.savefig (os.path.join (outdir, outlabel + 'dij_enrichment.svg'),
                  bbox_inches='tight', format='svg', transparent=True, dpi=600)
        plt.close ()
    else :
        plt.show ()


def compute_avg_sign_matrix (M, Ps=None, alpha=.1, nAA=21) :
    
    Llig, Lpos = M.shape
    nlig = int (Llig / nAA)
    npos = int (Lpos / nAA)
    
    # find the average sign
    Dsign = cp.deepcopy (M)
    Dsign[Dsign > 0] = 1
    Dsign[Dsign < 0] = -1
    if Ps is not None :
        Dsign[Ps > alpha] = np.nan

    avgSign = np.zeros ((nlig, npos))
    for i in range (nlig) :
        for j in range (npos) :
            avgSign[i,j] = np.nanmean (Dsign[i*nAA:(i*nAA + nAA),:][:,j*nAA:(j*nAA + nAA)])

    return avgSign


def multiple_test_correct (M) :
    
    nlig, npos = M.shape
    if np.sum (np.isnan (M)) == 0 :  
        fdrps = scipy.stats.false_discovery_control ( np.ndarray.flatten (M), method='bh')
        FP    = np.reshape (fdrps, (nlig, npos))
        
    return FP



@njit
def compute_D (coord) :
    
    L, ndim = coord.shape
    
    D = np.zeros ( (L, L) )
    for i in range (L) :
        for j in range (L) :
            D[i,j] = np.sqrt ( np.sum ( (coord[i,:] - coord[j,:])**2) )
            
    return D

def compute_avg_mat (A, n) :
    
    Nt   = A.shape[0]
    inc = int ( Nt / n)
    print (inc)    
    Av = np.zeros ((n, n)) * np.nan
    for i in range (n) :
        for j in range (i+1,n) :
            Av[i,j] = Av[j,i] = np.nanmean (A[(i*inc):(i*inc + inc),:][:,(j*inc):(j*inc + inc)])
            
    return Av


def plot_heatmap (A, sym=False, cmap='BrBG', sort=None) :
    if sort is not None :
        Acp = cp.deepcopy (A[np.argsort (sort),:][:,np.argsort (sort)])
    else :
        Acp = cp.deepcopy (A)
        
    if sym :
        maxval = np.max ( np.abs ([np.nanmin (A), np.nanmax (A)]))
        plt.imshow (Acp, cmap=cmap, vmin=-maxval, vmax=maxval)
    else :
        plt.imshow (Acp, cmap=cmap)

    plt.colorbar (shrink=0.7)
    plt.show ()



def stacked_heatmaps (xvals, yvals, yrange=(0,1), nreps=3, xlabel=None, ylabel=None, xlin=False, ylin=False) :

    plt.rcParams["figure.figsize"] = (8,4)
    mypinks  = seaborn.color_palette ('rocket', 10)


    fig, axs = plt.subplots (2, 3, height_ratios=[1.2,2], sharey='row', sharex=True, constrained_layout=True)

    if xlin :
        xbins = np.linspace (np.nanmin (xvals), np.nanmax (xvals), 15)
    else :
        xbins = np.logspace (np.log10 (np.nanmin (xvals)), np.log10 (np.nanmax (xvals)), 15)
    
    print (xbins)
    if ylin :
        ybins = np.linspace (yrange[0], yrange[1], 15)
    else :
        ybins = np.logspace (np.log10 (np.nanmin (yvals)), np.log10 (np.nanmax (yvals)), 15)

    minval = 1
    maxval = 0
    
    hist_list = list ()
    for r in range (nreps) :

        y  = np.ndarray.flatten (cp.deepcopy (xvals[:,:,r]))
        ps = np.ndarray.flatten (cp.deepcopy (yvals[:,:,r]))

        norm = 1
        hist, xedges, yedges = np.histogram2d (y[np.logical_and (~np.isnan (y), ~np.isnan (ps))],
                                               ps[np.logical_and (~np.isnan (y), ~np.isnan (ps))],
                                               bins=(xbins, ybins))
        
        hist = hist.T

        with np.errstate (divide='ignore', invalid='ignore'):  # suppress division by zero warnings
            hist *= norm / hist.sum(axis=0, keepdims=True)
            
        hist_list.append (hist)       

        minval = np.nanmin ( [minval, np.nanmin (hist[hist != 0])])
        maxval = np.nanmax ( [maxval, np.nanmax (hist[hist != 0])])
            
    # now plot
    for r in range (nreps) :          
            
        y  = np.ndarray.flatten (cp.deepcopy (xvals[:,:,r]))
        axs[0,r].hist (y, bins=xbins, color=mypinks[1])

        pcm = axs[1,r].pcolormesh (xedges, yedges, hist_list[r], cmap='BuPu',
                                   norm=mpl.colors.LogNorm (vmin=minval, vmax=maxval))
        if not xlin :
            axs[1,r].set_xscale ('log')
            axs[0,r].set_xscale ('log')

        if not ylin :
            axs[1,r].set_yscale ('log')
            axs[0,r].set_yscale ('log')
            
            
        
        
        axs[1,r].set_xlabel (xlabel)

    axs[0,0].set_ylabel (r'Count')
    axs[1,0].set_ylabel (ylabel)
    fig.colorbar (pcm, ax=axs[1,2], shrink=0.8, label='Fraction')

    plt.show()
   

# DELTE EVENTUALLY
def make_fpr_figure (P, Imat, ys, outlabel='', alpha=.1, ithres=0, binsize=50, save=False) :

    hit_list  = list ()
    tot_list  = list ()
    for sign in ['zero','positive','negative'] :
        Pvals   = cp.deepcopy (P)
        bigImat = cp.deepcopy (Imat)
        
        Hits    = np.zeros_like (Pvals)
        FPs     = np.zeros_like (Pvals)
        Tot_fp  = np.zeros_like (Pvals)
        Tot     = np.zeros_like (Pvals)
        
        # sort matrices by single effects
        Pval_sort    = cp.deepcopy (Pvals[np.argsort (ys),:][:,np.argsort (ys)])
        bigImat_sort = cp.deepcopy (bigImat[np.argsort (ys),:][:,np.argsort (ys)])

        if sign == 'zero' :
            # true positives
            Hits[np.logical_and (Pval_sort < alpha, np.abs (bigImat_sort) > ithres)] = 1
            Hits[np.isnan (Pval_sort)] = np.nan
            # total number non-contacts
            Tot[np.logical_and (~np.isnan (Pval_sort), np.abs (bigImat_sort) > ithres)] = 1

        elif sign == 'positive' :
            Hits[np.logical_and (Pval_sort < alpha, bigImat_sort > ithres)] = 1
            Hits[np.isnan (Pval_sort)] = np.nan
            Tot[np.logical_and (~np.isnan (Pval_sort), bigImat_sort > ithres)] = 1

        else :
            Hits[np.logical_and (Pval_sort < alpha, bigImat_sort < ithres)] = 1
            Hits[np.isnan (Pval_sort)] = np.nan
            Tot[np.logical_and (~np.isnan (Pval_sort), bigImat_sort < ithres)] = 1

        # store sign results
        hit_list.append (cp.deepcopy (Hits))
        tot_list.append (cp.deepcopy (Tot))
        
    # total false positives
    FPs[np.logical_and (Pval_sort < alpha, np.abs (bigImat_sort) <= ithres)] = 1
    FPs[np.isnan (Pval_sort)] = np.nan

    # total zeros
    Tot_fp = np.zeros_like (Pval_sort)
    Tot_fp[np.logical_and (~np.isnan (Pval_sort), np.abs (bigImat_sort) <= ithres)] = 1

    # now go through lists to compute power and false positives
    nvals = np.sum (~np.isnan (ys))
    bs    = binsize 
    nbins = int (nvals / bs)

    power_list = list ()
    for k in range (3) :
        Power = np.zeros ( (nbins, nbins) )
        Posit = np.zeros ( (nbins, nbins) )
        Totavg = np.zeros_like (Posit)
        for i in range (nbins) :
            for j in range (i,nbins) :
                num_ij = np.nansum (hit_list[k][i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs])
                den_ij = np.nansum (tot_list[k][i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs])
                Power[i,j] = Power[j,i] = num_ij / den_ij

                num_ij = np.nansum (FPs[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs])
                den_ij = np.nansum (Tot_fp[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs])
                Posit[i,j] = Posit[j,i] = num_ij / den_ij

                Totavg[i,j] = Totavg[j,i] = np.nanmean ( np.abs (bigImat_sort[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs]) > 0)
        power_list.append (cp.deepcopy (Power))

    # make figure
    plt.rcParams["figure.figsize"] = (5.5,.66*6.)

    fig, axs = plt.subplots (2, 2, sharex=True, sharey=True, constrained_layout=True)

    h1 = axs[0,0].imshow (power_list[0], cmap='rocket_r', vmin=0)
    fig.colorbar (h1, ax=axs[0,0], shrink=0.75, label='t.p.r.')

    h2 = axs[1,0].imshow (Posit, cmap='Blues', vmin=0) #norm=mpl.colors.LogNorm ())
    fig.colorbar (h2, ax=axs[1,0], shrink=0.75, label='f.p.r.')

    h11 = axs[0,1].imshow (power_list[1], cmap='rocket_r', vmin=0)
    fig.colorbar (h11, ax=axs[0,1], shrink=0.75, label='t.p.r.')

    h12 = axs[1,1].imshow (power_list[2], cmap='rocket_r', vmin=0)
    fig.colorbar (h12, ax=axs[1,1], shrink=0.75, label='t.p.r.')

    for i in range (2) :
        for j in range (2) :
            axs[i,j].set_xticks (np.arange (0,nbins,10), np.arange (0,nbins,10))

        axs[i,0].set_yticks (np.arange (0,nbins,10), np.arange (0,nbins,10))

    # label axes
    for i in range (2) :
        axs[i,0].set_ylabel (r'$\hat Y_j$ rank bin')
        axs[i,1].set_title ([r'$\lambda_{ij} < 0$',r'$\lambda_{ij} > 0$'][i], fontsize=12)
        axs[i,0].set_title ([r'$|\lambda_{ij}| > 0$',r'$\lambda_{ij} = 0$'][i], fontsize=12)

        axs[1,i].set_xlabel (r'$\hat Y_i$ rank bin')

    if save :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_sorted_pos_four.pdf'), bbox_inches='tight')
        plt.close ()
    else :
        plt.show () 

# DELETE EVENTUALLY
def make_fpr_figure_contacts (P, Imat, Dmat, ys, outlabel='', alpha=.1, ithres=0, binsize=50, save=False) :

    Pvals   = cp.deepcopy (P)
    bigImat = cp.deepcopy (Imat)
    
    # create sign matrix
    SignD   = cp.deepcopy (Dmat)
    SignD[Dmat < 0] = -1
    SignD[Dmat > 0] = 1
    SignD[Pvals > alpha] = np.nan
    
    Hits    = np.zeros_like (Pvals)
    FPs     = np.zeros_like (Pvals)
    Tot_fp  = np.zeros_like (Pvals)
    Tot     = np.zeros_like (Pvals)

    # sort matrices by single effects
    Pval_sort    = cp.deepcopy (Pvals[np.argsort (ys),:][:,np.argsort (ys)])
    bigImat_sort = cp.deepcopy (bigImat[np.argsort (ys),:][:,np.argsort (ys)])
    Sign_sort    = cp.deepcopy (SignD[np.argsort (ys),:][:,np.argsort (ys)])
    
    # true positives
    Hits[np.logical_and (Pval_sort < alpha, np.abs (bigImat_sort) > ithres)] = 1
    Hits[np.isnan (Pval_sort)] = np.nan
    # total number contacts
    Tot[np.logical_and (~np.isnan (Pval_sort), np.abs (bigImat_sort) > ithres)] = 1
        
    # total false positives
    FPs[np.logical_and (Pval_sort < alpha, np.abs (bigImat_sort) <= ithres)] = 1
    FPs[np.isnan (Pval_sort)] = np.nan
    # total zeros
    Tot_fp = np.zeros_like (Pval_sort)
    Tot_fp[np.logical_and (~np.isnan (Pval_sort), np.abs (bigImat_sort) <= ithres)] = 1

    # now compute power and false positives
    nvals = np.sum (~np.isnan (ys))
    bs    = binsize 
    nbins = int (nvals / bs)

    Power  = np.zeros ( (nbins, nbins) )
    Posit  = np.zeros ( (nbins, nbins) )
    Totavg = np.zeros_like (Posit)
    Totsum = np.zeros_like (Posit)
    Savg   = np.zeros_like (Posit)
    NumHits = np.zeros_like (Posit)
    for i in range (nbins) :
        for j in range (i,nbins) :
            num_ij = np.nansum (Hits[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs])
            den_ij = np.nansum (Tot[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs])
            Power[i,j] = Power[j,i] = num_ij / den_ij

            num_ij = np.nansum (FPs[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs])
            den_ij = np.nansum (Tot_fp[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs])
            Posit[i,j] = Posit[j,i] = num_ij / den_ij

            Totavg[i,j] = Totavg[j,i] = np.sum ( np.abs (bigImat_sort[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs]) > ithres)
            Totsum[i,j] = Totsum[j,i] = np.sum ( ~np.isnan (bigImat_sort[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs]) )
            
            Savg[i,j] = Savg[j,i] = np.nanmean ( Sign_sort[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs] )
            NumHits[i,j] = NumHits[j,i] = np.sum ( ~np.isnan( Sign_sort[i*bs:(i+1)*bs,:][:,j*bs:(j+1)*bs] ))
            
    # make figure
    plt.rcParams["figure.figsize"] = (4.5,.66*6.)

    fig, axs = plt.subplots (2, 2, sharex=True, sharey=True, constrained_layout=True)
   
    h0 = axs[0,0].imshow (Totavg / Totsum, cmap='gray_r')
    fig.colorbar (h0, ax=axs[0,0], shrink=0.75)#, label='Prop.') #label=r'$|\lambda_{ij}| > 0$')
    
    xs, ys = np.where (Totsum < 5)
    for k in range (len (xs)) :
        axs[0,0].add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                                edgecolor='black', linestyle='dotted', alpha=1., lw=1))

    
    h1 = axs[0,1].imshow (Power, cmap='rocket_r', vmin=0)
    fig.colorbar (h1, ax=axs[0,1], shrink=0.75)#, label='t.p.r.')

    h2 = axs[1,0].imshow (Posit, cmap='Blues', vmin=0) #norm=mpl.colors.LogNorm ())
    fig.colorbar (h2, ax=axs[1,0], shrink=0.75)#, label='f.p.r.')

    h11 = axs[1,1].imshow (Savg, cmap='Spectral', vmin=-1, vmax=1)
    fig.colorbar (h11, ax=axs[1,1], shrink=0.75)#, label=r'Avg. $sign (\hat D_{ij})$')
    xs, ys = np.where (NumHits < 10)
    for k in range (len (xs)) :
        axs[0,0].add_patch( Rectangle((ys[k]-0.5, xs[k]-.5), 1, 1, fill=False,
                                edgecolor='black', linestyle='dotted', alpha=1., lw=1))

    for i in range (2) :
        for j in range (2) :
            axs[i,j].set_xticks (np.arange (0,nbins,10), np.arange (0,nbins,10))

        axs[i,0].set_yticks (np.arange (0,nbins,10), np.arange (0,nbins,10))

    # label axes
    for i in range (2) :
        axs[i,0].set_ylabel (r'$\hat Y_j$ rank bin')
        axs[1,i].set_xlabel (r'$\hat Y_i$ rank bin')

    axs[0,0].set_title (r'Prop. $|\lambda_{ij}| > 0$', size=12)
    axs[0,1].set_title (r'True pos. rate', size=12)
    axs[1,0].set_title ('False pos. rate', size=12)
    axs[1,1].set_title (r'Avg. sign $\hat D_{ij}$', size=12)
        
    if save :
        plt.savefig (os.path.join (outdir, str (outlabel) + '_sorted_pos_four.pdf'), bbox_inches='tight')
        plt.close ()
    else :
        plt.show ()
