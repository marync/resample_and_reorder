import os
import sys
import pandas

# my code
sys.path.insert(1, '../code/')
from format_helper_functions import *
from rank_functions import compute_transformed_Rmat
from D_functions import compute_D_from_ranks


# Hardware
#-------------------------------------------------------------------------------
rootdir   = str (sys.argv[1])
indir     = str (sys.argv[2])
seed      = int (sys.argv[3])
sfactor   = int (sys.argv[4])
sigma     = float (sys.argv[5])
threshold = int (sys.argv[6])
two_state = str (sys.argv[7]) == 'True'
all_equal = str (sys.argv[8]) == 'True'
sample_range = str (sys.argv[9]) == 'True' # uniform lambda values

# reduce epistasis?
if len (sys.argv) >= 11 :
    theta = float (sys.argv[10])
else :
    theta = None 

ns = 0
if len (sys.argv) >= 12 :
    ns = float (sys.argv[11])
    #ns = 0.0007

print ('theta: ' + str (theta))
print ('ns: ' + str (ns))

# name for output
protein = 'sim'
# energy min and max
emin = -3.5
emax = 3.5

# minimum interaction strength
minInteraction = .1

# old argument for providing interaction matrix
imat_file = None

print (len (sys.argv))
print (sys.argv)
print ()

#if len (sys.argv) == 11 :
#    imat_file = str (sys.argv[10])
#else :
#    imat_file = None

# set rng
rng = np.random.default_rng (seed)

# add extra noise?
extra_noise = False


# Input and output
#-------------------------------------------------------------------------------
# output directory
outdir = rootdir
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

outdir = os.path.join (outdir, str (seed))
if not os.path.isdir (outdir) :
    os.makedirs (outdir)

# path to energy file
inenergies   = os.path.join (indir, 'GB1', 'energies.csv')
# "" distance matrix
distancefile = os.path.join (indir, 'GB1', 'GB1_distance_matrix.txt')

# single and double read count files
N0s_file = os.path.join (outdir, 'GB1_S_neutral.txt')
N0d_file = os.path.join (outdir, 'GB1_D_neutral.txt')


# Hard-coded parameters
#-------------------------------------------------------------------------------
# wild-type reads
N0_wt_obs = int (1759616 / sfactor)
N1_wt_obs = int (3041819 / sfactor)

# reads for all equal
C0s = 10000
C0d = 1000

# GB1 like parameters
npos = 55   # number of positions
nAA  = 20   # number of amino acids
#r_doubles = r_singles = 8.97
r_doubles = r_singles = 4
L    = npos*nAA

# simulation parameters
decay   = 2
sigma_e = .01
eta     = 1     # pseudocount
d_thres = 8

# threshold at which we resample the lambda values
gamma_threshold = 4

# wt sequence
wt_seq = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

# Read in data
#-------------------------------------------------------------------------------
# convert AA mutations to numeric
aadict = make_aa_dict ()

# energies
df_energies = pandas.read_csv (inenergies, delimiter=',')
df_energies['aa_num'] = df_energies['aa'].map (aadict)

energies_singles, energies_mat = format_energies_mat_from_df (df_energies.drop (0), npos,
                                                              ['efi','ebi'], nAA=20, offset=2)
gf_wt = df_energies['efi'].iloc[0]
gb_wt = df_energies['ebi'].iloc[0]
print ((gf_wt, gb_wt))

gf_singles = cp.deepcopy (energies_singles[:,0])#[~np.isnan (energies_singles[:,0])])
gb_singles = cp.deepcopy (energies_singles[:,1])#[~np.isnan (energies_singles[:,0])])

# physical map
Dmat = np.loadtxt (distancefile)
Dp   = Dmat[1:,1:] # match indices

# save energy information
np.savetxt (os.path.join (outdir, 'ordered_energies.txt'), energies_singles)


# Read in initial read counts 
#-------------------------------------------------------------------------------
N0s_obs = np.loadtxt (N0s_file)
N0d_obs = np.loadtxt (N0d_file)


# Convert gf and gb -> gb in two-state model 
#-------------------------------------------------------------------------------
# npos * nAA
L = len (gf_singles)
nmutants = int (L*(L-1) / 2)

# compute wild type energy
if two_state :
    print ('Computing two-state probabilities.')
    lambda_wt = np.log ( np.exp (gb_wt) * (np.exp (gf_wt) + 1) )
    #lambda_wt = 0.
    print ('Wildtype lambda: ' + str (lambda_wt))

    # compute the single effects
    #egt = (np.exp (gb_singles + gb_wt) * (np.exp (gf_singles + gf_wt) + 1))
    ewt = np.exp (gb_wt) * (np.exp (gf_wt) + 1)
    #ewt = 1.
    print ('Wildtype energy: ' + str (ewt))
    
    if sample_range :
        print ('Sampling from uniform.')
        # sample energy effects uniformly across interval
        gamma_singles = cp.deepcopy (gb_singles)
        gamma_singles[~np.isnan (gb_singles)] = rng.uniform (emin, emax, np.sum (~np.isnan (gb_singles)))
        egt = np.exp (gamma_singles)
    
    else :
        print ('Sampling from realistic distribution.')
        # compute the single effects
        egt = (np.exp (gb_singles + gb_wt) * (np.exp (gf_singles + gf_wt) + 1))
        ewt = np.exp (gb_wt) * (np.exp (gf_wt) + 1)
        print ('Wildtype energy: ' + str (ewt))
        gamma_singles = ( np.log (egt) - lambda_wt ) 

        # resample very deleterious mutants
        gamma_singles[gamma_singles > gamma_threshold] = rng.uniform (0,gamma_threshold, np.sum (gamma_singles > gamma_threshold))


    np.savetxt (os.path.join (outdir, 'sampled_single_lambdas.txt'), gamma_singles)


    # get double mutation lambda values
    bigS  = np.reshape (np.repeat (gamma_singles, L), (L,L))
    bigS += np.transpose (np.reshape (np.repeat (gamma_singles, L), (L,L)) )

    # set values to missing along diagonal; wt should already be missing
    for i in range (npos) :
        bigS[i*nAA:(i*nAA + nAA),:][:,i*nAA:(i*nAA + nAA)] = np.nan

    # calculate the probability of binding for all of the doubles
    p_doubles = 1 / (1. + np.exp (bigS + lambda_wt))
    # save energy information
    np.savetxt (os.path.join (outdir, 'p_doubles.txt'), p_doubles)
    p_doubles[np.isnan (p_doubles)] = 0.

    p_singles = 1. / (1. + np.exp (gamma_singles + lambda_wt))
    p_singles[np.isnan (p_singles)] = 0.

    # set scalar multiplier
    pwt = (1. / (1. + ewt))
    #r_doubles = r_singles = N1_wt / ((N0_wt + eta) * pwt) 

else :
    print ('Computing three-state probabilities.')
    # get double mutation lambda values
    bigS_f  = np.reshape (np.repeat (gf_singles, L), (L,L))
    bigS_f += np.transpose (np.reshape (np.repeat (gf_singles, L), (L,L)) )
    
    # get double mutation lambda values
    bigS_b  = np.reshape (np.repeat (gb_singles, L), (L,L))
    bigS_b += np.transpose (np.reshape (np.repeat (gb_singles, L), (L,L)) )
    
    # set values to missing along diagonal; wt should already be missing
    for i in range (npos) :
        bigS_f[i*nAA:(i*nAA + nAA),:][:,i*nAA:(i*nAA + nAA)] = np.nan
        bigS_b[i*nAA:(i*nAA + nAA),:][:,i*nAA:(i*nAA + nAA)] = np.nan
    
    # calculate the probability of binding for all of the doubles
    p_doubles = 1 / (1. + np.exp (bigS_b + gb_wt) * (np.exp (bigS_f + gf_wt) + 1.))
    p_doubles[np.isnan (p_doubles)] = 0.

    p_singles = 1. / (1. + np.exp (gb_singles + gb_wt) * (np.exp (gf_singles + gf_wt) + 1.))
    p_singles[np.isnan (p_singles)] = 0.

    # set scalar multiplier
    pwt = (1. / (1. + np.exp (gb_wt) * (np.exp (gf_wt) + 1)))
    
    # just for setting things to missing
    bigS = cp.deepcopy (bigS_f)

print ('Setting constant factor: ' + str (r_doubles))

# Sample read counts before and after selection
#-------------------------------------------------------------------------------
# sample reads before selection
#N0  = np.array (rng.lognormal ( mus_olson, vars_olson, L), dtype=int)
#N0d = np.array (rng.lognormal ( mud_olson, vard_olson, nmutants ), dtype=int) 
if all_equal : 
    print ('Sampling from equal reads.')
    # singles
    N0 = rng.poisson ( C0s / sfactor, size=len (N0s_obs))
    N0[N0s_obs == -1]  = 0

    # doubles
    N0dtmp = cp.deepcopy (N0d_obs)
    N0dtmp[N0d_obs != -1] = C0d / sfactor
    N0dtmp[N0d_obs == -1] = 0
    N0d = rng.poisson (N0dtmp[np.tril_indices (L, k=-1)])

    # wildtype
    N0_wt = rng.poisson (N0_wt_obs)

else :
    print ('Sampling from empirical distribution with scaling factor: ' + str (sfactor))
    # scale the entries, if below threshold put at threshold, add a pseudo count
    F0s = np.ceil (N0s_obs / sfactor)
    F0s[N0s_obs / sfactor < threshold] = threshold
    F0s[N0s_obs == -1] = -1

    F0dtmp = np.ceil (N0d_obs / sfactor)
    F0dtmp[N0d_obs / sfactor < threshold] = threshold
    F0dtmp[N0d_obs == -1] = -1
    F0d = cp.deepcopy (F0dtmp[np.triu_indices (L, k=1)])

    # sample cell counts
    C0s = rng.choice (F0s[F0s != -1], size=len (F0s), replace=True)
    C0d = rng.choice (F0d[F0d != -1], size=len (F0d), replace=True)

    # sample initial read counts
    N0    = rng.poisson ( C0s )
    N0d   = rng.poisson ( C0d )
    N0_wt = rng.poisson ( N0_wt_obs )
    

# set values to 0 which are below the read count threshold
N0d[N0d < threshold] = 0
N0[N0 < threshold]   = 0
    
# sample reads after selection
N1   = rng.poisson ( C0s * (p_singles * (1-ns) + ns) * r_singles) 
N1d  = rng.poisson ( C0d * ( np.ndarray.flatten (p_doubles[np.tril_indices (L, k=-1)]) * (1. - ns) + ns) * r_doubles )
N1wt = rng.poisson ( N0_wt * (pwt * (1.-ns) + ns) * r_singles)

# format as matrix
C0d_mat = np.zeros ( (L,L) )
C0d_mat[np.tril_indices (L, k=-1)] = cp.deepcopy (C0d)
C0d_mat += np.transpose (C0d_mat)
C0d_mat[np.isnan (bigS)] = np.nan

# format as matrix
N0d_mat = np.zeros ( (L,L) )
N0d_mat[np.tril_indices (L, k=-1)] = cp.deepcopy (N0d)
N0d_mat += np.transpose (N0d_mat)
N0d_mat[np.isnan (bigS)] = np.nan
N0d_mat[N0d_mat == 0] = np.nan

# post reads
N1d_mat = np.zeros ( (L,L) )
N1d_mat[np.tril_indices (L, k=-1)] = cp.deepcopy (N1d)
N1d_mat += np.transpose (N1d_mat)
N1d_mat[np.isnan (N0d_mat)]   = np.nan

# set to missing wts
N0_f = np.array (N0, dtype=float)
N1_f = np.array (N1, dtype=float)
N0_f[np.isnan (gf_singles)] = np.nan
N1_f[np.isnan (gf_singles)] = np.nan

# cell counts
if all_equal :
    C0_f = np.array (np.repeat (C0s, L), dtype=float)
else :
    C0_f = np.array (C0s, dtype=float)
# set to missing
C0_f[np.isnan (gf_singles)] = np.nan

np.savetxt (os.path.join (outdir, protein + '_S_neutral.txt'), N0_f) # fmt='%d')
np.savetxt (os.path.join (outdir, protein + '_D_neutral.txt'), N0d_mat) #, fmt='%d')
np.savetxt (os.path.join (outdir, protein + '_S_C0.txt'), C0_f) # fmt='%d')
np.savetxt (os.path.join (outdir, protein + '_D_C0.txt'), C0d_mat) #, fmt='%d')
np.savetxt (os.path.join (outdir, protein + '_S_selection.txt'), N1_f) #, fmt='%d')
np.savetxt (os.path.join (outdir, protein + '_D_selection_noep.txt'), N1d_mat) #, fmt='%d')
np.savetxt (os.path.join (outdir, protein + '_wt.txt'), [N0_wt, N1wt]) #, fmt='%d')

print ('WT N1: ' + str (N1wt))

# Compute single fitness
Y_singles = (N1_f + eta) / (N0_f + eta)

# Compute double fitness
Y_doubles = (N1d_mat + eta) / (N0d_mat + eta)


# Add some idiosyncratic epistasis if two-state!
#-------------------------------------------------------------------------------
if imat_file is None and two_state :

    # sample the individual coefficients
    bigImat = np.zeros ( (L, L) )
    for i in range (npos) :
        for j in range (i+1, npos) :
            if Dp[i,j] <= 5 :
                ep_ij       = np.reshape (rng.normal (0, scale=np.sqrt (sigma) * np.sqrt (np.exp (-Dp[i,j] / decay)), size=nAA**2), 
                                         (20,20))
                b_ij        = (2. * rng.binomial (1, .5) - 1.)
            else :
                ep_ij = b_ij = 0
    
            if extra_noise :
                ep_ij_noise = np.reshape (rng.normal (0, scale=np.sqrt (sigma_e), size=nAA**2), 
                                         (20,20))
                bigImat[(i*nAA):(i*nAA + nAA),:][:,(j*nAA):(j*nAA + nAA)] = (b_ij*( np.abs (ep_ij)) + ep_ij_noise)

            else :
                bigImat[(i*nAA):(i*nAA + nAA),:][:,(j*nAA):(j*nAA + nAA)] = b_ij*( np.abs (ep_ij))
        
        # fill diagonal blocks with nan
        bigImat[(i*nAA):(i*nAA + nAA),:][:,(i*nAA):(i*nAA + nAA)] = np.nan 
    
    bigImat += np.transpose (bigImat)
    #np.fill_diagonal (bigImat, np.nan)
    

elif two_state :
    bigImat = np.loadtxt ( imat_file )


if two_state :
    # set some coefficients to zero
    if theta is not None :
        print ('Setting coefficients to 0 with probability ' + str (theta))
        print (np.sum (bigImat == 0))
        print (np.sum (np.abs (bigImat) > 0))
        Zero = rng.binomial (1, 1-theta, bigImat.shape)
        Zero[np.triu_indices (L, k=0)] = 0
        Zero += np.transpose (Zero)
        bigImat *= Zero
        print (np.sum (bigImat == 0))
        print (np.sum (np.abs (bigImat) > 0))
    else :
        print ()
        print ('Keeping epistatic coefficients as is')
        print ()
    
    
    # set very small coefficients to 0
    bigImat[np.abs (bigImat) < minInteraction] = 0.
    print (np.sum ( np.abs (bigImat) > 0))
    
    
    # write out epistatic file
    np.savetxt (os.path.join (outdir, 'bigImat.txt'), bigImat) 

# Resample the read counts, now including i.e.
#-------------------------------------------------------------------------------

if two_state :
    print ('Adding epistasis.')
    p_doubles = 1 / (1. + np.exp (bigS + bigImat + lambda_wt))
    p_doubles[np.isnan (p_doubles)] = 0.
   
    np.savetxt ( os.path.join (outdir, 'lambda_doubles_epistasis.txt'), bigS + bigImat ) 
    # read counts
    N1d  = rng.poisson ( C0d * ( np.ndarray.flatten (p_doubles[np.tril_indices (L, k=-1)]) * (1.-ns) + ns) * r_doubles )
    
    N1d_mat = np.zeros ( (L,L) )
    N1d_mat[np.tril_indices (L, k=-1)] = cp.deepcopy (N1d)
    N1d_mat += np.transpose (N1d_mat)
    N1d_mat[np.isnan (N0d_mat)] = np.nan

    # set values back to na to write out
    p_doubles[np.isnan (N0d_mat)] = np.nan

    np.savetxt (os.path.join (outdir, 'p_doubles_epistasis.txt'), p_doubles) #, fmt='%d')

# Save everything (if three state then just resave same sample)
#-------------------------------------------------------------------------------
# save to file
np.savetxt (os.path.join (outdir, protein + '_D_selection.txt'), N1d_mat) #, fmt='%d')

# Compute double fitness
Y_doubles = (N1d_mat + eta) / (N0d_mat + eta)

np.savetxt (os.path.join (outdir, protein + '_Y_doubles.txt'), Y_doubles)
np.savetxt (os.path.join (outdir, protein + '_Y_singles.txt'), Y_singles)

# compute the rank matrix
M = np.nanmax (np.sum (~np.isnan (Y_doubles), axis=0)) - 1
Rmat    = compute_transformed_Rmat (Y_doubles, M=M)
Dmat    = compute_D_from_ranks (Y_singles, Rmat, M=M) 

np.savetxt (os.path.join (outdir, protein + '_Rmat.txt'), Rmat)
np.savetxt (os.path.join (outdir, protein + '_Dmat.txt'), Dmat)

