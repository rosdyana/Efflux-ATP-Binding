import os
import sys
import subprocess

fastadir = sys.argv[1]
nrdb = "/home/semmy/blastLinux/blast-2.2.26/db/nr"
iteration = 3
numthread = 4

for fastafile in os.listdir(fastadir):
    print("processing {}".format(fastafile))
    outfile = "pssm/{}.pssm".format(fastafile.split(".")[0])
    infile = "{}/fasta/{}".format(os.getcwd(), fastafile)
    subprocess.call(
        'psiblast -db {} -num_iterations {} -num_threads {} -in_msa {} -out_ascii_pssm {}'.format(
            nrdb, iteration, numthread, infile, outfile), shell=True)
