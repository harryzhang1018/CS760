import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define a function to calculate n and errn for a given dataset and decision tree
def calculate_n_errn(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    n = clf.tree_.node_count
    # r = export_text(clf)
    # print(r)
    y_pred = clf.predict(X_test)
    errn = 1 - accuracy_score(y_test, y_pred)
    return n, errn

# Load the datasets
#data_files = ["./hw02/data/D32.txt", "./hw02/data/D128.txt", "./hw02/data/D512.txt", "./hw02/data/D2048.txt", "./hw02/data/D8192.txt"]

# Initialize lists to store n and errn values for each dataset
n_values = []
errn_values = []
dataset = np.loadtxt("./hw02/data/Dbig.txt", delimiter=' ')
np.random.shuffle(dataset)
data = dataset[0:8192,:]
X_test = dataset[8192:,0:2]
y_test = dataset[8192:,2]
training_size = [32, 128, 512, 2048, 8192]
# Loop through each dataset
for data_file in training_size:
    # Load the dataset from the file (replace with your data loading code)
    # X = np.genfromtxt(data_file, delimiter=' ')[:,0:2]
    # y = np.genfromtxt(data_file, delimiter=' ')[:,2]
    X_train = data[0:data_file,0:2]
    y_train = data[0:data_file,2]
    #y = y.reshape(-1, 1)
    #print(X.shape, y.shape)
    # Split the dataset into training and testing sets (adjust test_size as needed)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

    # Calculate n and errn for this dataset
    n, errn = calculate_n_errn(X_train, X_test, y_train, y_test)
    print(n, errn)
    # Store the n and errn values for this dataset
    n_values.append(n)
    errn_values.append(errn)

# Plot n vs. errn for each dataset
plt.figure(figsize=(12, 6))
for i, data_file in enumerate(training_size):
    plt.scatter(n_values[i], errn_values[i], label=f'Dataset {i+1}: '+str(training_size[i]) + ' training instances')
plt.plot(n_values, errn_values, color='blue')
plt.title('Number of Nodes vs. Error Rate for Decision Trees (Unlimited Depth)')
plt.xlabel('Number of Nodes (n)')
plt.ylabel('Error Rate (errn)')
plt.legend()
plt.savefig('./hw02/data/p3_sklearn.png')
plt.show()
