import numpy as np
import matplotlib.pyplot as plt

f_pos_rate = [0,0/4, 1/4, 2/4, 1]
t_pos_rate = [0,2/6, 4/6, 1, 1]

plt.figure(figsize=(6, 6))
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curve', fontsize=18)
plt.plot(f_pos_rate, t_pos_rate, 'r-o')
plt.show()