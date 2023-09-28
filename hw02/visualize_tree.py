import numpy as np
import matplotlib.pyplot as plt

## scatter plot for D1 and D2

def readfiles(filename):
    data = np.loadtxt(filename, delimiter=' ')
    feature = data[:,0:2]
    instances = data[:,2]
    return feature, instances

D1_feature, D1_instances = readfiles('./hw02/data/D1.txt')
D2_feature, D2_instances = readfiles('./hw02/data/D2.txt')

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(D1_feature[:,0][D1_instances==1], D1_feature[:,1][D1_instances==1], c='blue', marker='o', label='class 1')
plt.scatter(D1_feature[:,0][D1_instances==0], D1_feature[:,1][D1_instances==0], c='red', marker='o', label='class 0')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.subplot(1,2,2)
plt.scatter(D2_feature[:,0][D2_instances==1], D2_feature[:,1][D2_instances==1], c='blue', marker='o', label='class 1')
plt.scatter(D2_feature[:,0][D2_instances==0], D2_feature[:,1][D2_instances==0], c='red', marker='o', label='class 0')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.tight_layout()
# plt.show()

plt.figure(figsize=(5,5))

plt.scatter([0 ,1],[0,1], c='blue', marker='o', label='class 1')
plt.scatter([0],[1], c='red', marker='o', label='class 0')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()