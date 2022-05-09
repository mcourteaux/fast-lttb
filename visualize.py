import numpy as np
import matplotlib.pyplot as plt

ref = np.genfromtxt("input.csv", delimiter=',')

#data1 = np.genfromtxt("ds_1m_1024.csv", delimiter=',')
#data2 = np.genfromtxt("ds_1m_10240.csv", delimiter=',')

data1 = np.genfromtxt("output_scalar.csv", delimiter=',')
data2 = np.genfromtxt("output_simd.csv", delimiter=',')

plt.plot(ref[:,0], ref[:,1])
plt.plot(data1[:,0], data1[:,1])
plt.plot(data2[:,0], data2[:,1])
plt.show()
