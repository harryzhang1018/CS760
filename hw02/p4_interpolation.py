import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

def calculate_error(x, y, poly):
    y_pre = poly(x)
    error = np.mean( (y - y_pre)**2 )
    error = np.log10(error)
    return error

# define training dataset
upper_bound = 0
lower_bound = 2*np.pi
total_size = 100

##define training and testing set
sample_size = 90
testing_size = total_size - sample_size

x_train = np.linspace(lower_bound, upper_bound, sample_size)
y_train = np.sin(x_train)

x_test = np.linspace(lower_bound, upper_bound, testing_size)
y_test = np.sin(x_test)

print(x_train.shape, y_train.shape)
# implement lagrange interpolation
poly = lagrange(x_train, y_train)
# print(Polynomial(poly.coef[::5]).coef)
y_pre = poly(x_train)

training_error = calculate_error(x_train, y_train, poly)
testing_error = calculate_error(x_test, y_test, poly)
print("training error: ", training_error)
print("testing error: ", testing_error)

#plot the resutls
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(x_test, y_test, '--', label='training data')
plt.plot(x_train, y_pre, '-', label='prediction')
plt.xlabel('x')
plt.ylabel('y ')
plt.legend()
plt.show()

# Experiment with varying standard deviations of Gaussian noise (ε)
noise_std_values = [0.01, 0.1, 0.5, 1.0, 5.0]
train_er = []
test_er = []
for noise_std in noise_std_values:
    x_train = np.linspace(lower_bound, upper_bound, sample_size)+np.random.normal(0, noise_std, sample_size)
    y_train = np.sin(x_train)
    x_test = np.linspace(lower_bound, upper_bound, testing_size)+np.random.normal(0, noise_std, testing_size)
    y_test = np.sin(x_test)
    poly = lagrange(x_train, y_train)
    train_error = calculate_error(x_train, y_train, poly)
    test_error = calculate_error(x_test, y_test, poly)
    train_er.append(train_error)
    test_er.append(test_error)
    
    print(f"Noise Std: {noise_std:.2f}")
    print("Train Error:", train_error)
    print("Test Error:", test_error)
    print("------------------------")
plt.figure(figsize=(12, 6))
plt.plot(noise_std_values, train_er, '--', label='training error')
plt.plot(noise_std_values, test_er, '-', label='testing error')
plt.legend()
plt.xlabel('noise std')
plt.ylabel('error in log scale ($log_{10}$)')
plt.show()