import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import roc_curve,roc_auc_score

k = 5

#Read the training dataset
training_data = np.genfromtxt('./hw03/data/hw3Data/emails.csv', delimiter=',')[1:, 1:]
#print(training_data.shape)


data_test = training_data[4000:, :]
print(data_test.shape)
data_training = training_data[0:4000, :]

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
    
    #calculate confidence rate (the number of the label one / k)
    confidence_rate = label_counts[1]/k
    
    return most_common_label,confidence_rate

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize hyperparameters
learning_rate = 0.0001
num_iterations = 2000

def LogisticRegressionMetrics(X_train,y_train,X_test,y_test):


    # Initialize the parameter vector theta
    theta = np.zeros(X_train.shape[1])

    # Perform gradient descent to optimize parameters
    for iteration in range(num_iterations):
        z = np.dot(X_train, theta)
        h = sigmoid(z)
        gradient = np.dot(X_train.T, (h - y_train)) 
        theta -= learning_rate * gradient

    # Make predictions on the testing data
    z_test = np.dot(X_test, theta)
    h_test = sigmoid(z_test)
    #y_pred = (h_test >= 0.5).astype(int)
    confidence_rate = h_test
    return confidence_rate

print("Running 5-NN and get the prediction")
predicted_labels = np.array([k_nearest_neighbor(data_training[:,0:3001], test_point,k) for test_point in data_test[:, 0:3001]])
print(predicted_labels.shape)
fpr_5nn, tpr_5nn, thresholds = roc_curve(data_test[:,-1], predicted_labels[:,1])
roc_auc_5nn = roc_auc_score(data_test[:,-1], predicted_labels[:,1])
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_5nn, tpr_5nn, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_5nn:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for 5-NN')
# plt.legend(loc='lower right')
# plt.show()

print("Running Logistic Regression and get the prediction")
confidence = LogisticRegressionMetrics(data_training[:,0:3000],data_training[:,-1],data_test[:,0:3000],data_test[:,-1])
print(confidence.shape)
fpr, tpr, thresholds = roc_curve(data_test[:,-1], confidence)
roc_auc = roc_auc_score(data_test[:,-1],confidence)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'LogisticRegression-ROC curve (AUC = {roc_auc:.2f})')
plt.plot(fpr_5nn, tpr_5nn, lw=2, label=f'5-NN-ROC curve (AUC = {roc_auc_5nn:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()