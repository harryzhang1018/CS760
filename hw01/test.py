import numpy as np
import matplotlib.pyplot as plt


N = 100

mean = [1, -1]
cov_mat = [[2, 0], [0, 2]]
two_dim_normal = np.random.multivariate_normal(mean, cov_mat,N)

plt.figure()
plt.title('2D Gaussian Distribution')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.scatter(two_dim_normal[:,0], two_dim_normal[:,1],marker='o')
plt.savefig('./hw01/images/2d_gaussian.png')



mean_1 = [5, 0]
cov_mat_1 = [[1, 0.25], [0.25, 1]]
two_dim_normal_1 = np.random.multivariate_normal(mean_1, cov_mat_1,N)
# print(two_dim_normal_1)
mean_2 = [-5, 0]
cov_mat_2 = [[1, -0.25], [-0.25, 1]]
two_dim_normal_2 = np.random.multivariate_normal(mean_2, cov_mat_2,N)

sum_data = 0.3*two_dim_normal_1 + 0.7*two_dim_normal_2

plt.figure()
plt.title('2D Mixture Distribution')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.scatter(sum_data[:,0], sum_data[:,1],marker='o')
plt.savefig('./hw01/images/2d_sumdata.png')