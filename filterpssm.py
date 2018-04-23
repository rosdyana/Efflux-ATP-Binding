# /**
#  * @author [Rosdyana Kusuma]
#  * @email [rosdyana.kusuma@gmail.com]
#  * @create date 2018-04-19 03:09:15
#  * @modify date 2018-04-23 03:19:41
#  * @desc [Tool to filter pssm from similarity list]
# */

import os
import sys
import pandas as pd
from shutil import copyfile

finput = sys.argv[1]
outdir = "similar{}pssm".format(finput[-2:])

if not os.path.exists(outdir):
    os.makedirs(outdir)

dinput = pd.read_csv(finput, header=None)
for i, v in enumerate(dinput[0]):
    print("copying {}".format(v))
    origin = "{}/pssm/{}.pssm".format(os.getcwd(), v)
    destination = "{}/{}/{}.pssm".format(os.getcwd(), outdir, v)
    try:
        copyfile(origin, destination)
    except Exception as e:
        print("{} not found, try the alternative".format(v))
        print("copying alternate {}".format(dinput[1][i]))
        origin = "{}/pssm/{}.pssm".format(os.getcwd(), dinput[1][i])
        destination = "{}/{}/{}.pssm".format(os.getcwd(), outdir, dinput[1][i])
        copyfile(origin, destination)
