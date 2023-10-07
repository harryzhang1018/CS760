import numpy as np
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

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize hyperparameters
learning_rate = 0.01
num_iterations = 1000

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
    y_pred = (h_test >= 0.5).astype(int)

    # Calculate accuracy, precision, and recall
    correct_predictions = (y_pred == y_test).astype(int)
    accuracy = np.mean(correct_predictions)

    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Print the results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

print("fold 1")
LogisticRegressionMetrics(fold1_training[:,0:3000],fold1_training[:,-1],fold1_test[:,0:3000],fold1_test[:,-1])
print("fold 2")
LogisticRegressionMetrics(fold2_training[:,0:3000],fold2_training[:,-1],fold2_test[:,0:3000],fold2_test[:,-1])
print("fold 3")
LogisticRegressionMetrics(fold3_training[:,0:3000],fold3_training[:,-1],fold3_test[:,0:3000],fold3_test[:,-1])
print("fold 4")
LogisticRegressionMetrics(fold4_training[:,0:3000],fold4_training[:,-1],fold4_test[:,0:3000],fold4_test[:,-1])
print("fold 5")
LogisticRegressionMetrics(fold5_training[:,0:3000],fold5_training[:,-1],fold5_test[:,0:3000],fold5_test[:,-1])
