import numpy as np
import matplotlib.pyplot as plt

## for k means accuracy
k_acc = [0.8233, 0.71, 0.64, 0.55, 0.456]
k_obj = [335.8238544564644,490.19454372324583,911.9733098424105,1693.5530205158198,3351.091513424485]
gmm_acc = [0.8166666666666667, 0.7133333333333334,0.55,0.53,0.5033333333333333]
gmm_obj = [950.337682681918,1074.3875850720924,1235.2078254071828,1413.522651223593,1615.4962246025816]
sigma = [ 0.5, 1, 2, 4, 8]
plt.figure()
plt.plot(sigma, k_acc, 'ro-', label='k-means')
plt.plot(sigma, gmm_acc, 'bo-', label='GMM')
plt.xlabel('sigma')
plt.ylabel('accuracy')
plt.title('Accuracy vs. sigma')
plt.legend()
plt.show()

plt.figure()
plt.plot(sigma, k_obj, 'ro-', label='k-means')
plt.plot(sigma, gmm_obj, 'bo-', label='GMM')
plt.xlabel('sigma')
plt.ylabel('objective')
plt.title('Objective vs. sigma')
plt.legend()
plt.show()