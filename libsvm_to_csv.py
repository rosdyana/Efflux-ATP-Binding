# /**
#  * @author [Rosdyana Kusuma]
#  * @email [rosdyana.kusuma@gmail.com]
#  * @create date 2018-04-23 03:23:17
#  * @modify date 2018-04-23 03:23:17
#  * @desc [Convert LIBSVM to CSV]
# */
import os
import sys


def libsvmToCsv(filename, OutputFile):
    #onlyfiles = [f for f in listdir(scrFolder) if isfile(join(scrFolder, f))]
    # for filename in onlyfiles:
    print(filename + " is processing")
    f1 = open(filename, "r")
    f2 = open(OutputFile, "a")  # thay đuôi libsvm thành csv
    lines = f1.readlines()
    for line in lines:
        classtype = line[0]  # lấy giá trị class
        line = line[1:-1]  # ko lấy class vì đã lưu ở trên rồi, ko lấy enter
        values = line.split()

        # ghi thông tin lên file đích
        for pair in values:
            colonPosition = pair.find(":")
            valueToWriteToFile = pair[colonPosition + 1:]
            f2.write(valueToWriteToFile + ", ")
        # ghi giá trị class ở cuối dòng
        f2.write(classtype)
        f2.write("\n")  # hết 1 dòng thì xuống hàng
    f1.close()
    f2.close()


dinput = sys.argv[1]
douput = "similar{}csv".format(dinput[7:-6])

if not os.path.exists(douput):
    os.makedirs(douput)

for fileLibsvm in os.listdir(dinput):
    onlyName = fileLibsvm.split(".")[0]
    inputFile = "{}/{}".format(dinput, fileLibsvm)
    outputFile = "{}/{}.csv".format(douput, onlyName)
    libsvmToCsv(inputFile, outputFile)
