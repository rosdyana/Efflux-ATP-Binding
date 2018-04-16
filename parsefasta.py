import os
import sys

finput = sys.argv[1]
with open(finput) as f:
    # for line in f:
    #     if ">" in line:
    #         print(line.split("|")[1])
    #     else:
    lines = f.read()
    lines = lines.split('>')
    lines = ['>' + x for x in lines[1:]]
    for x in lines:
        # use this variable to create the new file
        file_name = x.split("|")[1]
        fil = open('fasta/' + file_name + '.fasta', 'w+')
        fil.write(x)
        fil.close()
