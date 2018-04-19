# /**
#  * @author [Rosdyana Kusuma]
#  * @email [rosdyana.kusuma@gmail.com]
#  * @create date 2018-04-19 03:09:15
#  * @modify date 2018-04-19 03:09:26
#  * @desc [Tool to filter pssm from similarity list]
# */

import os
import sys
import pandas as pd
from shutil import copyfile

finput = sys.argv[1]
outdir = sys.argv[2]

dinput = pd.read_csv(finput, header=None)
for i in dinput[0]:
    print("copying {}".format(i))
    origin = "{}/pssm/{}.pssm".format(os.getcwd(),i)
    destination = "{}/{}/{}.pssm".format(os.getcwd(),outdir,i)
    try:
        copyfile(origin,destination)
    except Exception as e:
        print(e)