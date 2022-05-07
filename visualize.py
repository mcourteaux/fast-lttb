import numpy as np
import matplotlib.pyplot as plt

data1 = np.genfromtxt("ds_1m_1024.csv", delimiter=',')
data2 = np.genfromtxt("ds_1m_10240.csv", delimiter=',')
plt.plot(data1[:,0], data1[:,1])
plt.plot(data2[:,0], data2[:,1])
plt.show()
