import os
import sys
import subprocess

fastadir = sys.argv[1]
nrdb = "/home/semmy/blastLinux/blast-2.2.26/db/nr"
iteration = 3
numthread = 4

for fastafile in os.listdir(fastadir):
    outfile = "pssm/{}.pssm".format(fastafile.split(".")[0])
    subprocess.call(
        'psiblast -db {} -num_iterations {} -num_threads {} -in_msa {} -out_ascii_pssm {}'.format(
            nrdb, iteration, numthread, fastafile, outfile), shell=True)
