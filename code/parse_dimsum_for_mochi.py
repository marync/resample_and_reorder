import os
import sys
import pandas

datadir = sys.argv[1]
outdir  = sys.argv[2]

outname = os.path.join (outdir, 'all_fitness.txt')
outf    = open (outname, 'w')

colnames = ['aa_seq', 'Nham_aa', 'WT', 'fitness', 'sigma']

# wt fitness file
wtdata   = pandas.read_csv (os.path.join (datadir, 'fitness_wildtype.txt'), delimiter=' ')
doubdata = pandas.read_csv (os.path.join (datadir, 'fitness_doubles.txt'), delimiter=' ')
singdata = pandas.read_csv (os.path.join (datadir, 'fitness_singles.txt'), delimiter=' ')
print (wtdata.columns)

# write out header line
outf.write ( ('\t').join (colnames) + '\n')

# wt info
for index, row in wtdata.iterrows():
    wt = row[colnames]
    outf.write ( ('\t').join ([str (x) for x in wt]) + '\n' )

colnames = ['aa_seq', 'Nham_aa', 'fitness_uncorr', 'sigma_uncorr']

# wt info
for index, row in doubdata.iterrows():
    vals = list (row[colnames])
    vals.insert (2, '')
    outf.write ( ('\t').join ([str (x) for x in vals]) + '\n' )


colnames = ['aa_seq', 'Nham_aa', 'fitness', 'sigma']

for index, row in singdata.iterrows():
    vals = list (row[colnames])
    vals.insert (2, '')
    outf.write ( ('\t').join ([str (x) for x in vals]) + '\n' )

outf.close ()







