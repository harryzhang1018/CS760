import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the training dataset
training_data = np.loadtxt('./hw03/data/hw3Data/D2z.txt')

# Extract features and labels from the training data
X_train = training_data[:, :-1]  # Features
y_train = training_data[:, -1]   # Labels

# Step 2: Create a 2D grid of test points
x_range = np.arange(-2, 2.1, 0.1)
y_range = np.arange(-2, 2.1, 0.1)
xx, yy = np.meshgrid(x_range, y_range)
grid_points = np.column_stack((xx.ravel(), yy.ravel()))

# Step 3: Implement 1-nearest neighbor classification using Euclidean distance for grid points
def one_nearest_neighbor(train_data, test_point):
    distances = np.linalg.norm(train_data - test_point, axis=1)
    nearest_neighbor_index = np.argmin(distances)
    return y_train[nearest_neighbor_index]

predicted_labels = np.array([one_nearest_neighbor(X_train, test_point) for test_point in grid_points])

# Step 4: Visualize the predictions and overlay the training set
plt.figure(figsize=(10, 8))

# Plot the grid points and their predictions
plt.scatter(grid_points[:, 0], grid_points[:, 1], c=predicted_labels, cmap='coolwarm', marker='x', label='Grid Points')
plt.colorbar(label='Predicted Labels', ticks=np.unique(predicted_labels))
    
# Overlay the training set
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='jet', marker='o', label='Training Points')
plt.legend(loc='best')

# Set axis limits and labels
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Show the plot
plt.title('1-NN Classification with Euclidean Distance')
plt.show()
