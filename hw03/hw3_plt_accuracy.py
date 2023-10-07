import matplotlib.pyplot as plt
import numpy as np
k = [1,3,5, 7, 10]
accuracy_1 = np.array([0.825,0.853,0.862,0.851,0.775])
accuracy_3 = np.array([0.846,0.85,0.856,0.88,0.773])
accuracy_5 = np.array([0.837,0.852,0.871,0.869,0.78])
accuracy_7 = np.array([0.837,0.861,0.875,0.874,0.779])
accuracy_10 = np.array([0.849,0.865,0.875,0.88,0.781])
accuracy = [np.mean(accuracy_1),np.mean(accuracy_3),np.mean(accuracy_5),np.mean(accuracy_7),np.mean(accuracy_10)]
plt.figure(figsize=(10, 6))
plt.plot(k, accuracy, 'r-o')
plt.xlabel('k', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy vs k', fontsize=18)
plt.show()