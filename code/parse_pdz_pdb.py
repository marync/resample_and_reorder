# Parse PDZ pdb file
filename = 'data/pdb/5heb.pdb'
f = open (filename, 'r')

outf = open ('data/fuzzy/pdz_pos.csv', 'w')
for line_raw in f.readlines () :
    line = line_raw.strip ().split ()
    if line[0] == 'ATOM' : #and line[4] in ['E','F'] :
        #print (line)
        if line[2][0] in ['C', 'O', 'N'] :
            if int (line[5]) >= 303 and int (line[5]) <= 402 :
                outf.write ((',').join (line[5:9]))
                outf.write ('\n')
                #outf.write (line_raw)
            if int (line[5]) >= 3 and int (line[5]) <= 9 :
                outf.write ((',').join (line[5:9]))
                outf.write ('\n')
                #outf.write ((',').join (line[6:9]))
                #outf.write ('\n')
                #outf.write (line_raw)


