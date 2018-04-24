# /**
#  * @author [Rosdyana Kusuma]
#  * @email [rosdyana.kusuma@gmail.com]
#  * @create date 2018-04-23 03:27:02
#  * @modify date 2018-04-23 03:27:02
#  * @desc [generate dataset]
# */

import os
import sys
import subprocess
import math
import random

dinput = sys.argv[1]
percentage = float(sys.argv[2])

src_files = (os.listdir(dinput))

fullpath = "{}/{}".format(os.getcwd(), dinput)


def valid_path(dir_path, filename):
    full_path = os.path.join(dir_path, filename)
    return os.path.isfile(full_path)


def merge_files(outputfile, listfiles):
    with open(outputfile, 'w') as outfile:
        for fname in listfiles:
            with open("{}/{}".format(fullpath, fname)) as infile:
                for line in infile:
                    outfile.write(line)


files = [os.path.join(dinput, f) for f in src_files if valid_path(dinput, f)]

training_list = []
validation_list = []
independent_list = []

path, dirs, files = os.walk(dinput).__next__()
file_count = len(files)
training_num = math.ceil(file_count * percentage)
independent_num = (file_count - training_num) // 3
validation_num = file_count - training_num - independent_num
print("{} = {} - {} - {}".format(file_count,
                                 training_num, validation_num, independent_num))

for i, v in enumerate(files):
    if i < independent_num:
        independent_list.append(v)
    elif i > independent_num and i <= validation_num + independent_num:
        validation_list.append(v)
    else:
        training_list.append(v)

trainingoutput = "{}training.csv".format(dinput[:9])
validationoutput = "{}validation.csv".format(dinput[:9])
independentoutput = "{}independent.csv".format(dinput[:9])

merge_files(trainingoutput, training_list)
merge_files(validationoutput, validation_list)
merge_files(independentoutput, independent_list)
