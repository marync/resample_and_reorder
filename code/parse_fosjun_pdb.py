# Script to parse pdb file for Fos-Jun
filename = 'data/pdb/1fos.pdb'
f = open (filename, 'r')

outf = open ('data/fosjun/fosjun_pos.csv', 'w')
for line_raw in f.readlines () :
    line = line_raw.strip ().split ()
    if line[0] == 'ATOM' and line[4] in ['E','F'] :
        if line[2][0] in ['C', 'O', 'N'] :
            if int (line[5]) >= 162 and int (line[5]) <= 193 :
                outf.write ((',').join (line[5:9]))
                outf.write ('\n')
                #outf.write (line_raw)
            if int (line[5]) >= 286 and int (line[5]) <= 317 :
                outf.write ((',').join (line[5:9]))
                outf.write ('\n')
                #outf.write ((',').join (line[6:9]))
                #outf.write ('\n')
                #outf.write (line_raw)


