import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('Accuracy.txt', header = None)
array = data.to_numpy()
tableAccuracy = array.reshape(20, 1500)

data = pd.read_csv('MSE.txt', header = None)
array = data.to_numpy()
tableMSE = array.reshape(20, 1500)

dfAccuracy = pd.DataFrame(tableAccuracy)
dfMSE = pd.DataFrame(tableMSE)

AccPlot = dfAccuracy[:].mean()
AccPlot = AccPlot.to_numpy()

MSEPlot = dfMSE[:].mean()
MSEPlot = MSEPlot.to_numpy()

itr = np.zeros(1500)

for i in range(1500):
	itr[i] = int(i)


plt.plot(itr, AccPlot)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.legend(['Train'], loc='upper left')
plt.show()

plt.plot(itr, MSEPlot)
plt.title('Mean square error')
plt.ylabel('Mean square error')
plt.xlabel('Iteration')
plt.legend(['Train'], loc='upper right')
plt.show()