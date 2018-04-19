import os
import sys
import subprocess

fastadir = sys.argv[1]
blastpgp = "/home/semmy/blastLinux/blast-2.2.26/bin/blastpgp"
nrdb = "/home/semmy/blastLinux/blast-2.2.26/db/nr"
iteration = 2
numthread = 4

for fastafile in os.listdir(fastadir):
    print("processing {}".format(fastafile))
    outfile = "pssm/{}.pssm".format(fastafile.split(".")[0])
    infile = "{}/fasta/{}".format(os.getcwd(), fastafile)
    subprocess.call(
        '{} -d {} -j {} -a {} -i {} -Q {}'.format(
            blastpgp, nrdb, iteration, numthread, infile, outfile), shell=True)
