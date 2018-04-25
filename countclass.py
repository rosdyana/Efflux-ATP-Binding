import sys
import pandas as pd

finput = sys.argv[1]

data = pd.read_csv(finput, header=None)
count0 = 0
count1 = 0
# count1 = data[data[340] == 1].count()
# count0 = data[data[340] == 0].count()
for i, v in enumerate(data[340]):
    if v == 0:
        count0 += 1
    elif v == 1:
        count1 += 1
print("num of 1 = {}\nnum of 0 = {}".format(count1, count0))
