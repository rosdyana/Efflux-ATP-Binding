# /**
#  * @author [Rosdyana Kusuma]
#  * @email [rosdyana.kusuma@gmail.com]
#  * @create date 2018-04-19 04:05:18
#  * @modify date 2018-04-19 04:05:18
#  * @desc [generate position of binding site in sequence]
# */
import json
import pandas as pd


def countThis(begin, end):
    list = []
    for i in range(int(begin), int(end) + 1):
        list.append(i)
    return list


datadict = {}
data = pd.read_csv("poslist", sep=",")
for i, v in enumerate(data['Nucleotide binding']):
    ds = v.split()
    vlen = len(ds)
    proteinname = data['Entry'][i]
    # print(proteinname)
    finallist = []
    if vlen >= 12:
        list1 = countThis(ds[1], ds[2])
        list2 = countThis(ds[7], ds[8])
        finallist = list1 + list2
    elif vlen == 10:
        list1 = countThis(ds[1], ds[2])
        list2 = countThis(ds[6], ds[7])
        finallist = list1 + list2
    elif vlen <= 5:
        finallist = countThis(ds[1], ds[2])
    # print(finallist)
    datadict[proteinname] = finallist

outfile = "poslist.json"
with open(outfile, 'w') as fp:
    json.dump(datadict, fp, indent=4, sort_keys=True)
