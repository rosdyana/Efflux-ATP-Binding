# /**
#  * @author [Rosdyana Kusuma]
#  * @email [rosdyana.kusuma@gmail.com]
#  * @create date 2018-04-19 04:03:20
#  * @modify date 2018-04-23 03:19:21
#  * @desc [convert pssm file to libsvm format]
# */
import json
from pprint import pprint
import os
import sys

AMINO = ["A", "R", "N", "D", "C", "Q", "E", "G", "H",
         "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

zero_padding = ['0', '0', '0', '0', '0', '0', '0', '0', '0',
                '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

datajson = json.load(open('poslist.json'))


def readPSSM(filename):
    listOfList = []
    f = open(filename, "r")
    lines = f.readlines()[3:-6]
    for line in lines:
        # split by space
        temp = line[:-1].split(" ")
        # remove all ''
        temp = list(filter(lambda x: x != '', temp))
        # if temp[1] in AMINO:
        listOfList.append(temp[2:22])
    return listOfList


def generateDatasetWithWindowSize(pssmFile, windowSize, resultFile):
    print(pssmFile, " is processing")
    listOfList = readPSSM(pssmFile)
    listOfListWithZeroPadding = []

    numOfPadding = int((windowSize - 1) / 2)
    # zero padding at the beginning of pssm list of list
    for i in range(numOfPadding):
        listOfListWithZeroPadding.append(zero_padding)
    # next copy value after zero padding
    for l in listOfList:
        listOfListWithZeroPadding.append(l)
    # zero padding at the end of pssm list of list
    for i in range(numOfPadding):
        listOfListWithZeroPadding.append(zero_padding)

    f_result = open(resultFile, "a")
    proteinName = pssmFile.split("/")[1].split(".")[0]
    posData = datajson[proteinName]

    length = len(listOfListWithZeroPadding)
    start = 0
    end = start + windowSize - 1
    i = 0
    print("length of pssm list ", length)
    cont = True
    while i < length and cont:
        listToWrite = []
        #print("start ",start, " end ",end)
        classType = ""
        for j in range(start, end + 1):
            if j == (end - numOfPadding):
                if (j - numOfPadding + 1) in posData:
                    classType = "1"
                else:
                    classType = "0"
            for k in listOfListWithZeroPadding[j]:
                listToWrite.append(k)

        featureNum = 1
        f_result.write(classType + " ")
        for m in listToWrite:
            f_result.write(str(featureNum) + ":" + str(m) + " ")
            featureNum = featureNum + 1
        f_result.write("\n")

        i = i + 1
        start = start + 1
        end = start + windowSize - 1
        if start >= length or end >= length:
            cont = False
    f_result.close()


dinput = sys.argv[1]
windowSize = int(sys.argv[2])
doutput = "similar{}libsvm".format(dinput[7:-4])

if not os.path.exists(doutput):
    os.makedirs(doutput)

for pssmfile in os.listdir(dinput):
    inputFile = "{}/{}".format(dinput, pssmfile)
    resultFile = "{}/{}.libsvm".format(doutput, pssmfile.split(".")[0])
    generateDatasetWithWindowSize(inputFile, windowSize, resultFile)
