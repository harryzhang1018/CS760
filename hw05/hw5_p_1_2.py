import numpy as np
import matplotlib.pyplot as plt

## generate 300 data points
import numpy as np

def initialize_parameters(data, num_clusters):
    n_samples, n_features = data.shape
    np.random.seed(0)
    
    # Initialize cluster weights, means, and covariances randomly
    cluster_weights = np.random.rand(num_clusters)
    cluster_weights /= cluster_weights.sum()
    cluster_means = data[np.random.choice(n_samples, num_clusters, replace=False)]
    cluster_covariances = [np.cov(data.T) for _ in range(num_clusters)]

    return cluster_weights, cluster_means, cluster_covariances

def gaussian_pdf(x, mean, covariance):
    n = len(mean)
    det_cov = np.linalg.det(covariance)
    inv_cov = np.linalg.inv(covariance)
    x_mean = x - mean
    exponent = -0.5 * np.dot(x_mean, np.dot(inv_cov, x_mean))
    coefficient = 1.0 / (np.sqrt((2 * np.pi) ** n * det_cov))
    return coefficient * np.exp(exponent)

def negative_log_likelihood(data, cluster_weights, cluster_means, cluster_covariances):
    num_clusters = len(cluster_weights)
    n_samples = data.shape[0]
    log_likelihood = 0.0

    for j in range(n_samples):
        sample_likelihood = 0.0
        for i in range(num_clusters):
            sample_likelihood += cluster_weights[i] * gaussian_pdf(data[j], cluster_means[i], cluster_covariances[i])
        log_likelihood += np.log(sample_likelihood)

    return -log_likelihood

def expectation_step(data, cluster_weights, cluster_means, cluster_covariances):
    num_clusters = len(cluster_weights)
    n_samples, n_features = data.shape
    responsibilities = np.zeros((n_samples, num_clusters))

    for i in range(num_clusters):
        for j in range(n_samples):
            responsibilities[j, i] = cluster_weights[i] * gaussian_pdf(data[j], cluster_means[i], cluster_covariances[i])
    
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    return responsibilities

def maximization_step(data, responsibilities):
    num_clusters = responsibilities.shape[1]
    n_samples, n_features = data.shape

    cluster_weights = responsibilities.mean(axis=0)
    cluster_means = np.dot(responsibilities.T, data) / responsibilities.sum(axis=0)[:, np.newaxis]
    cluster_covariances = []

    for i in range(num_clusters):
        weighted_data = data - cluster_means[i]
        covariance = np.dot(responsibilities[:, i] * weighted_data.T, weighted_data) / responsibilities[:, i].sum()
        cluster_covariances.append(covariance)

    return cluster_weights, cluster_means, cluster_covariances

def predict_labels(data, cluster_weights, cluster_means, cluster_covariances):
    num_clusters = len(cluster_weights)
    n_samples = data.shape[0]
    predicted_labels = np.zeros(n_samples, dtype=int)

    for j in range(n_samples):
        # Calculate the responsibility for each cluster
        responsibilities = np.zeros(num_clusters)
        for i in range(num_clusters):
            responsibilities[i] = cluster_weights[i] * gaussian_pdf(data[j], cluster_means[i], cluster_covariances[i])

        # Assign the data point to the cluster with the highest responsibility
        predicted_labels[j] = np.argmax(responsibilities)

    return predicted_labels

def gaussian_mixture_model(data, num_clusters, max_iterations=100, tol=1e-6):
    cluster_weights, cluster_means, cluster_covariances = initialize_parameters(data, num_clusters)

    for iteration in range(max_iterations):
        old_cluster_weights, old_cluster_means, old_cluster_covariances = (
            cluster_weights.copy(), cluster_means.copy(), cluster_covariances.copy())

        responsibilities = expectation_step(data, cluster_weights, cluster_means, cluster_covariances)
        cluster_weights, cluster_means, cluster_covariances = maximization_step(data, responsibilities)

        # Check for convergence using the negative log likelihood
        log_likelihood = negative_log_likelihood(data, cluster_weights, cluster_means, cluster_covariances)
        if iteration > 0 and np.abs(log_likelihood - old_log_likelihood) < tol:
            break

        old_log_likelihood = log_likelihood

    # Return the predicted labels
    predicted_labels = predict_labels(data, cluster_weights, cluster_means, cluster_covariances)

    return cluster_weights, cluster_means, cluster_covariances, predicted_labels,log_likelihood

def k_means_clustering(data, num_clusters, max_iterations=100):
    """
    Perform K-means clustering on a dataset without using third-party libraries.

    Parameters:
    - data: A (n_samples, n_features) NumPy array containing the data.
    - num_clusters: The number of clusters to create.
    - max_iterations: Maximum number of iterations for the algorithm.

    Returns:
    - cluster_centers: An array containing the coordinates of the cluster centers.
    - cluster_labels: An array of cluster labels for each data point.
    """
    # Initialize cluster centers randomly
    np.random.seed(0)
    initial_centers_indices = np.random.choice(data.shape[0], num_clusters, replace=False)
    cluster_centers = data[initial_centers_indices]

    for _ in range(max_iterations):
        # Assign each data point to the nearest cluster
        distances = np.linalg.norm(data[:, np.newaxis] - cluster_centers, axis=2)
        cluster_labels = np.argmin(distances, axis=1)
        # if _ == 0:
        #     print("distance",distances)
        #     print("cluster labels",cluster_labels)

        # Update cluster centers
        new_cluster_centers = np.array([data[cluster_labels == i].mean(axis=0) for i in range(num_clusters)])

        # Check for convergence
        if np.all(cluster_centers == new_cluster_centers):
            print("Converged after {} iterations.".format(_ + 1))
            break

        cluster_centers = new_cluster_centers

    return cluster_centers, cluster_labels, distances**2

def evl_acc(means, centroids, labels, true_labels):
    mapping = {}
    for i in range(len(centroids)):
        min_dist = np.inf
        for j in range(len(means)):
            dist = np.linalg.norm(centroids[i] - means[j])
            if dist < min_dist:
                min_dist = dist
                mapping[i] = j
    # Convert all the labels based on this mapping
    for i in range(len(labels)):
        labels[i] = mapping[labels[i]]
 
    # Compute the accuracy
    acc = 0
    for i in range(len(labels)):
        if labels[i] == true_labels[i]:
            acc += 1
 
    return acc / len(labels)

# mean and covariance matrix
sigma = 0.5
print("sigma", sigma)
# data set A
mean_a =np.array( [-1, -1])
cov_a = np.array( [[2, 0.5], [0.5, 1]] )* sigma
p_a = np.random.multivariate_normal(mean_a, cov_a, 100)
#data set B
mean_b = np.array([1, -1])
cov_b = np.array([[1, -0.5], [-0.5, 2]]) * sigma
p_b = np.random.multivariate_normal(mean_b, cov_b, 100)
#data set C
mean_c = np.array([0, 1])
cov_c = np.array([[1, 0], [0, 2]]) * sigma
p_c = np.random.multivariate_normal(mean_c, cov_c, 100)
# combine data set
data = np.concatenate((p_a, p_b, p_c), axis=0)
true_labels = np.concatenate((np.zeros(100), np.ones(100), np.ones(100) * 2))
# print(true_labels)

centers, labels, obj = k_means_clustering(data, 3)
min_distances = np.min(obj, axis=1, keepdims=True)
k_obj = np.sum(min_distances)
means = np.vstack((mean_a, mean_b, mean_c))
k_means_acc = evl_acc(means, centers, labels, true_labels)
# print(centers)
# print(labels)
print("k means accuracy", k_means_acc)
print("k means objective", k_obj)
# cluster_weights, cluster_means, cluster_covariances, gmm_labels , objective= gaussian_mixture_model(data, 3)
# print(cluster_means)
# # print(cluster_covariances)
# gmm_means_acc = evl_acc(means, cluster_means, labels, true_labels)
# print(gmm_means_acc)
# print(objective)