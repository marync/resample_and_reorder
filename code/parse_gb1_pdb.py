filename = 'data/pdb/2j52.pdb'
f = open (filename, 'r')

outf = open ('data/GB1/gb1_heavy_pos.csv', 'w')
for line_raw in f.readlines () :
    line = line_raw.strip ().split ()
    if line[0] == 'ATOM' :
        if line[2][0] in ['C', 'O', 'N'] :
            #print (line[2])
            #print ()
            outf.write ((',').join (line[5:9]))
            outf.write ('\n')
        #else :
        #    print ('No')
        #    print (line[2])
        #    print ()


