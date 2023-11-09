import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def buggy_pca(X, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = np.dot(X, Vt.T[:, :d])
    V_d = Vt.T[:, :d]
    X_reconstructed = np.dot(Z, V_d.T)
    return Z, V_d, X_reconstructed

def demeaned_pca(X, d):
    mean = np.mean(X, axis=0)
    X_demeaned = X - mean
    U, S, Vt = np.linalg.svd(X_demeaned, full_matrices=False)
    Z = np.dot(X_demeaned, Vt.T[:, :d])
    V_d = Vt.T[:, :d]
    X_reconstructed = np.dot(Z, V_d.T) + mean
    return Z, V_d, X_reconstructed

def normalized_pca(X, d):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X - mean) / std_dev
    U, S, Vt = np.linalg.svd(X_normalized, full_matrices=False)
    Z = np.dot(X_normalized, Vt.T[:, :d])
    V_d = Vt.T[:, :d]
    X_reconstructed = (np.dot(Z, V_d.T) * std_dev) + mean
    return Z, V_d, X_reconstructed

def dro(data, d):
    n, D = data.shape
 
    # Step 1: Find the value for b in the optimal solution
    b = np.mean(data, axis=0)
 
    # Step 2: Define Y = X - b
    Y = data - b
 
    # Step 3: Take the Singular Value Decomposition of Y
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
 
    # Get the diagonal elements of S and also return those
    S_diag = S
 
    # Truncate the SVD components to d dimensions
    U = U[:, :d]
    S = np.diag(S[:d])
    Vt = Vt[:d, :]
 
    # Step 4: Map the truncated SVD components back to the original problem
    At = np.dot(S, Vt)
    Z = U
 
    # Calculate the d-dimensional representations
    d_dim_representations = Z  # Each column of Z is a d-dimensional representation
 
    # Calculate the reconstructions in D dimensions
    reconstructions = Z @ At + b
 
    return d_dim_representations, At, b,reconstructions

# # Generate a 2D dataset for testing
# np.random.seed(0)
print("For 2D dataset:")
data_2d = np.loadtxt('./hw04/data2D.csv', delimiter=',')

# Buggy PCA
U_buggy, Vt_buggy, recon_buggy = buggy_pca(data_2d,1)
recon_buggy_err = np.sum((data_2d - recon_buggy)**2)
print("Reconstruction error for buggy PCA: {}".format(recon_buggy_err))
# Demeaned PCA
U_demeaned, Vt_demeaned, recon_demeaned = demeaned_pca(data_2d,1)
recon_demeaned_err = np.sum((data_2d - recon_demeaned)**2)
print("Reconstruction error for demeaned PCA: {}".format(recon_demeaned_err))
# Normalized PCA
U_normalized, Vt_normalized, recon_normalized = normalized_pca(data_2d,1)
recon_normalized_err = np.sum((data_2d - recon_normalized)**2)
print("Reconstruction error for normalized PCA: {}".format(recon_normalized_err))
d_dim_representations, At, b, recon_dro= dro(data_2d,1)
recon_dro_err = np.sum((data_2d - recon_dro)**2)
print("Reconstruction error for DRO: {}".format(recon_dro_err))
# Plot original and reconstructed points
plt.figure(figsize=(12, 4))
plt.subplot(141)
plt.scatter(data_2d[:, 0], data_2d[:, 1], label='Original')
plt.scatter(recon_buggy[:, 0], recon_buggy[:, 1], label='Reconstructed')
plt.title('Buggy PCA')
plt.legend()

plt.subplot(142)
plt.scatter(data_2d[:, 0], data_2d[:, 1], label='Original')
plt.scatter(recon_demeaned[:, 0], recon_demeaned[:, 1], label='Reconstructed')
plt.title('Demeaned PCA')
plt.legend()

plt.subplot(143)
plt.scatter(data_2d[:, 0], data_2d[:, 1], label='Original')
plt.scatter(recon_normalized[:, 0], recon_normalized[:, 1], label='Reconstructed')
plt.title('Normalized PCA')
plt.legend()

plt.subplot(144)
plt.scatter(data_2d[:, 0], data_2d[:, 1], label='Original')
plt.scatter(recon_dro[:, 0], recon_dro[:, 1], label='Reconstructed')
plt.title('DRO')
plt.legend()
plt.show()

### doing the 1000D dataset
data_1000d = np.loadtxt('./hw04/data1000D.csv', delimiter=',')
# X_1000d = data_1000d - np.mean(data_1000d, axis=0)
## implement SVD and find the number of non zero singular values
U, S, Vt = np.linalg.svd(data_1000d, full_matrices=False)
# Assuming you have calculated singular values in S for DRO
singular_values = S
# Plot singular values
plt.figure()
plt.plot(range(1, len(singular_values) + 1), singular_values, marker='o', linestyle='-')
plt.xlabel("Principal Component")
plt.ylabel("Singular Value")
plt.title("Singular Value Spectrum")
plt.grid(True)
plt.show()

d_1k = 31
print("For 1000D dataset:")
# Buggy PCA
U_buggy, Vt_buggy, recon_buggy = buggy_pca(data_1000d,d_1k)
recon_buggy_err = np.sum((data_1000d - recon_buggy)**2)
print("Reconstruction error for buggy PCA: {}".format(recon_buggy_err))
# Demeaned PCA
U_demeaned, Vt_demeaned, recon_demeaned = demeaned_pca(data_1000d,d_1k)
recon_demeaned_err = np.sum((data_1000d - recon_demeaned)**2)
print("Reconstruction error for demeaned PCA: {}".format(recon_demeaned_err))
# Normalized PCA
U_normalized, Vt_normalized, recon_normalized = normalized_pca(data_1000d,d_1k)
recon_normalized_err = np.sum((data_1000d - recon_normalized)**2)
print("Reconstruction error for normalized PCA: {}".format(recon_normalized_err))
d_dim_representations, At, b, recon_dro= dro(data_1000d,d_1k)
recon_dro_err = np.sum((data_1000d - recon_dro)**2)
print("Reconstruction error for DRO: {}".format(recon_dro_err))

