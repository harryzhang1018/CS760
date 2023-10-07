import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

k = 10

#Read the training dataset
training_data = np.genfromtxt('./hw03/data/hw3Data/emails.csv', delimiter=',')[1:, 1:]
#print(training_data.shape)

# split data into 5 folds each has training and testing sets
fold1_test = training_data[0:1000, :]
fold1_training = training_data[1000:, :]

fold2_test = training_data[1000:2000, :]
fold2_training = np.concatenate((training_data[0:1000, :], training_data[2000:, :]), axis=0)

fold3_test = training_data[2000:3000, :]
fold3_training = np.concatenate((training_data[0:2000, :], training_data[3000:, :]), axis=0)

fold4_test = training_data[3000:4000, :]
fold4_training = np.concatenate((training_data[0:3000, :], training_data[4000:, :]), axis=0)

fold5_test = training_data[4000:, :]
fold5_training = training_data[0:4000, :]

# implement 1-nearest neighbor classification using Euclidean distance for grid points

def one_nearest_neighbor(train_data, test_point):
    distances = np.linalg.norm(train_data[:,0:3000] - test_point[0:3000], axis=1)
    nearest_neighbor_index = np.argmin(distances)
    return train_data[nearest_neighbor_index, -1]

def k_nearest_neighbor(train_data, test_point, k):
    distances = np.linalg.norm(train_data[:, 0:3000] - test_point[0:3000], axis=1)
    # Get the indices of the k smallest distances using argsort
    nearest_neighbor_indices = np.argsort(distances)[:k]
    # Get the labels of the k nearest neighbors
    nearest_neighbor_labels = train_data[nearest_neighbor_indices, -1]
    
    # Count the occurrences of each label
    label_counts = Counter(nearest_neighbor_labels)
    
    # Find the label with the highest count
    most_common_label = label_counts.most_common(1)[0][0]
    
    return most_common_label

def calculate_metrics(predicted_labels, actual_labels):
    """
    Calculate accuracy, precision, and recall.

    Parameters:
    - predicted_labels: Numpy array of predicted labels (1D array)
    - actual_labels: Numpy array of actual labels (1D array)

    Returns:
    - accuracy: Accuracy of the predictions
    - precision: Precision of the predictions
    - recall: Recall of the predictions
    """
    # Calculate True Positives (TP), True Negatives (TN),
    # False Positives (FP), and False Negatives (FN)
    tp = np.sum((predicted_labels == 1) & (actual_labels == 1))
    tn = np.sum((predicted_labels == 0) & (actual_labels == 0))
    fp = np.sum((predicted_labels == 1) & (actual_labels == 0))
    fn = np.sum((predicted_labels == 0) & (actual_labels == 1))

    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Calculate recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall

print("fold 1")
predicted_labels_1 = np.array([k_nearest_neighbor(fold1_training[:,0:3001], test_point,k) for test_point in fold1_test[:, 0:3001]])
# print(predicted_labels.shape)
accuracy, precision, recall = calculate_metrics(predicted_labels_1, fold1_test[:,-1])
print(accuracy)
print("fold 2")
predicted_labels_2 = np.array([k_nearest_neighbor(fold2_training[:,0:3001], test_point,k) for test_point in fold2_test[:, 0:3001]])
# print(predicted_labels.shape)
accuracy, precision, recall = calculate_metrics(predicted_labels_2, fold2_test[:,-1])
print(accuracy)
print("fold 3")
predicted_labels_3 = np.array([k_nearest_neighbor(fold3_training[:,0:3001], test_point,k) for test_point in fold3_test[:, 0:3001]])
accuracy, precision, recall = calculate_metrics(predicted_labels_3, fold3_test[:,-1])
print(accuracy)
print("fold 4")
predicted_labels_4 = np.array([k_nearest_neighbor(fold4_training[:,0:3001], test_point,k) for test_point in fold4_test[:, 0:3001]])
accuracy, precision, recall = calculate_metrics(predicted_labels_4, fold4_test[:,-1])
print(accuracy)
print("fold 5")
predicted_labels_5 = np.array([k_nearest_neighbor(fold5_training[:,0:3001], test_point,k) for test_point in fold5_test[:, 0:3001]])
accuracy, precision, recall = calculate_metrics(predicted_labels_5, fold5_test[:,-1])
print(accuracy)