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
testing_list = []
validation_list = []

path, dirs, files = os.walk(dinput).__next__()
file_count = len(files)
training_num = math.ceil(file_count * percentage)
validation_num = (file_count - training_num) // 3
testing_num = file_count - training_num - validation_num
print("{} = {} - {} - {}".format(file_count,
                                 training_num, testing_num, validation_num))

for i, v in enumerate(files):
    if i < validation_num:
        validation_list.append(v)
    elif i > validation_num and i <= testing_num + validation_num:
        testing_list.append(v)
    else:
        training_list.append(v)

trainingoutput = "{}training.csv".format(dinput[:9])
testingoutput = "{}testing.csv".format(dinput[:9])
validationoutput = "{}validation.csv".format(dinput[:9])

merge_files(trainingoutput, training_list)
merge_files(testingoutput, testing_list)
merge_files(validationoutput, validation_list)
