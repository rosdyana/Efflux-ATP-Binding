import json
from pprint import pprint
import os

AMINO = ["A", "R", "N", "D", "C", "Q", "E", "G", "H",
         "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

zero_padding = ['0', '0', '0', '0', '0', '0', '0', '0', '0',
                '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

datajson = json.load(open('poslist.json'))
# for i, v in enumerate(datajson['Q9UNQ0']):
#     print(v)


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
    # --> ['155', '157', '184', '184', '188', '189', '193', '193', '217', '218']
    # posSeg = posData[:-1].split()
    # posList = []
    # i = 0
    # print("length of posSeg ", len(posSeg))
    # while i < len(posSeg):
    #     start = int(posSeg[i])
    #     end = int(posSeg[i + 1])
    #     for j in range(start, end + 1):
    #         posList.append(j)
    #     i = i + 2
    # print(posList) # --> [155, 156, 157, 184, 188, 189, 193, 217, 218]

    # generate dataset according to window size
    # chu y padding roi thi index thay doi
    length = len(listOfListWithZeroPadding)
    start = 0
    end = start + windowSize - 1
    i = 0
    print("length of pssm list ", length)
    cont = True
    while i < length and cont:  # duyet theo list cac dong pssm
        listToWrite = []
        #print("start ",start, " end ",end)
        classType = ""
        for j in range(start, end + 1):
            if j == (end - numOfPadding):  # xet dong chinh giữa
                if (j - numOfPadding + 1) in posData:
                    classType = "1"
                else:
                    classType = "0"
            for k in listOfListWithZeroPadding[j]:
                listToWrite.append(k)

        featureNum = 1
        # ghi du lieu ra file ket qua
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
    # Ket thuc vong lap while
    f_result.close()


dinput = "similar30pssm"
windowSize = 17
doutput = "similar30libsvm"
for pssmfile in os.listdir(dinput):
    inputFile = "{}/{}".format(dinput, pssmfile)
    resultFile = "{}/{}.libsvm".format(doutput, pssmfile.split(".")[0])
    generateDatasetWithWindowSize(inputFile, windowSize, resultFile)
