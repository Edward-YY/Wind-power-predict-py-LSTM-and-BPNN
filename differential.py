import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statistics import mean

path = 'dataset/newaichi15min.csv'
df = pd.read_csv(path, header=0, index_col=None)
data = pd.read_csv('dataset/result.csv',header = 0, index_col= 0)
dataValue = data.values
lstmPower = dataValue[:,1]
bpPower = dataValue[:,2]
actPower = dataValue[:,0]

def makeDiffer(ary):
    n = len(ary)
    differList = list()
    for i in range(1,n):
        differ = round((ary[i] - ary[i-1]), 2)
        differList.append(differ)
        differArray = np.array(differList)
    return differArray

lstmDiffer = makeDiffer(lstmPower)
bpDiffer = makeDiffer(bpPower)
actDiffer = makeDiffer(actPower)

lstmMax = lstmDiffer.max()
lstmMin = lstmDiffer.min()
bpMax = bpDiffer.max()
bpMin = bpDiffer.min()
actMax = actDiffer.max()
actMin = actDiffer.min()

lstmAverage = np.sum(lstmDiffer) / len(lstmDiffer)
bpAverage = np.sum(bpDiffer) / len(bpDiffer)
actAverage = np.average(actDiffer)

lstmToAct = lstmDiffer - actDiffer
bpToAct = bpDiffer - actDiffer

plt.plot(actDiffer,label = "Actual")
plt.plot(lstmDiffer,label = "lstm")
plt.plot(bpDiffer,label = "BPNN3")

xtick = np.arange(1,97,16)
igfont = {'family':'IPAexGothic'}
plt.xlabel('Time Point(15min)',**igfont)
plt.ylabel('Power Fluctuation(MW/15min)',**igfont)
plt.legend()
plt.show()

lstmRmse = np.sqrt(mean_squared_error(actDiffer,lstmDiffer))
bpRmse = np.sqrt(mean_squared_error(actDiffer,bpDiffer))

print("RMSE of LSTM differential is: {}\n RMSE of BPNN differential is: {}".format(lstmRmse,bpRmse))
print("lstmMax:{},lstmMin{}\nbpMax:{},bpMin:{}\nactMax:{},actMin:{}".format(\
    lstmMax,lstmMin,bpMax,bpMin,actMax,actMin))
print("lstmAverage:{}\nbpAverage:{}\nactAverage:{}".format\
          (lstmAverage,bpAverage,actAverage))