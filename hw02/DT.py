import numpy as np
from collections import Counter
import graphviz
import pandas as pd
from typing import Callable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class DecisionTreeNode:
    def __init__(self):
        self.feature_idx = None  # Index of the feature for the split
        self.threshold = None    # Threshold value for the split
        self.left = None         # Left subtree
        self.right = None        # Right subtree
        self.is_leaf = False     # Is this node a leaf?
        self.predicted_class = None  # Predicted class for leaf nodes
        self.node_count = 1  # Initialize with 1 for the current node

def entropy(y):
    """Calculate the entropy of a binary class distribution."""
    p1 = np.sum(y) / len(y)
    p0 = 1 - p1
    if p0 == 0 or p1 == 0:
        return 0
    return -p0 * np.log2(p0) - p1 * np.log2(p1)

def information_gain(y, y_left, y_right):
    """Calculate information gain."""
    H_parent = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)
    N = len(y)
    N_left = len(y_left)
    N_right = len(y_right)
    gain = H_parent - (N_left / N) * H_left - (N_right / N) * H_right
    return gain

def information_gain_ratio(y, y_left, y_right):
    """Calculate information gain."""
    H_parent = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)
    N = len(y)
    N_left = len(y_left)
    N_right = len(y_right)
    gain = H_parent - (N_left / N) * H_left - (N_right / N) * H_right
    
    # Calculate the intrinsic information
    intrinsic_info = -((N_left / N) * np.log2(N_left / N) + (N_right / N) * np.log2(N_right / N))

    # Calculate the gain ratio
    if intrinsic_info == 0:
        return 0  # Avoid division by zero
    gain_ratio = gain / intrinsic_info

    return gain_ratio

def find_best_split(X, y, candidate_splits):
    """Find the best split for the dataset among candidate splits."""
    best_gain = -1
    best_split = None
    for j, c in candidate_splits:
        X_left = X[X[:, j] >= c]
        y_left = y[X[:, j] >= c]
        X_right = X[X[:, j] < c]
        y_right = y[X[:, j] < c]

        if len(y_left) == 0 or len(y_right) == 0:
            # print("zero split gain",j,c)
            # print("info gain: ",information_gain(y, y_left, y_right))
            continue  # Skip empty splits

        gain = information_gain_ratio(y, y_left, y_right)
        #print("for split: ",(j,c),"gain: ",gain)
        if gain == 0:
            return None
        else:
            if gain > best_gain:
                best_gain = gain
                best_split = (j, c)

    return best_split

def make_subtree(X, y):
    node = DecisionTreeNode()
    node.node_count += 1
    #print("here")
    # Stopping criteria
    if len(set(y)) == 1:
        node.is_leaf = True
        node.predicted_class = y[0]
        return node

    if len(X) == 0:
        node.is_leaf = True
        node.predicted_class = 1  # fPredict class 1 when there's no majority class
        return node

    candidate_splits = []
    for j in range(X.shape[1]):
        for c in np.unique(X[:, j]):
            candidate_splits.append((j, c))
    #print("candidate split: ",candidate_splits)
    best_split = find_best_split(X, y, candidate_splits)
    #print("best split: ",best_split)
    if best_split is None:
        print("no best split")
        node.is_leaf = True
        count = Counter(y)
        node.predicted_class = count.most_common(1)[0][0] # Predict the majority class
        print(node.predicted_class)
    else:
        node.feature_idx, node.threshold = best_split
        X_left = X[X[:, node.feature_idx] >= node.threshold]
        y_left = y[X[:, node.feature_idx] >= node.threshold]
        X_right = X[X[:, node.feature_idx] < node.threshold]
        y_right = y[X[:, node.feature_idx] < node.threshold]

        node.left = make_subtree(X_left, y_left)
        node.right = make_subtree(X_right, y_right)

    return node

def decision_tree_classifier(X_train, y_train):
    root = make_subtree(X_train, y_train)
    return root

def predict(tree, X_new):
    if isinstance(X_new, pd.Series):
        X_new = X_new.to_numpy()
    # Traverse the decision tree to predict the class label
    current_node = tree
    while not current_node.is_leaf:
        feature_idx = current_node.feature_idx
        threshold = current_node.threshold
        if X_new[feature_idx] >= threshold:
            current_node = current_node.left
        else:
            current_node = current_node.right
    return current_node.predicted_class


def readfiles(filename):
    data = np.loadtxt(filename, delimiter=' ')
    feature = data[:,0:2]
    instances = data[:,2]
    return feature, instances

def tree_to_dot(node, dot, feature_names=None, class_names=None):
    if node is None:
        return

    if node.is_leaf:
        label = f"Class {node.predicted_class}"
    else:
        label = f"{feature_names[node.feature_idx]}  >  Threshold {node.threshold:.4f}"

    dot.node(str(id(node)), label=label)

    if node.left is not None:
        dot.edge(str(id(node)), str(id(node.left)), label="True")
        tree_to_dot(node.left, dot, feature_names, class_names)

    if node.right is not None:
        dot.edge(str(id(node)), str(id(node.right)), label="False")
        tree_to_dot(node.right, dot, feature_names, class_names)

def count_nodes(node):
    if node is None:
        return 0, 0

    if node.is_leaf:
        return 0, 1  # Leaf node

    left_internal, left_leaves = count_nodes(node.left)
    right_internal, right_leaves = count_nodes(node.right)

    # Current node is internal, so increment internal count
    internal_nodes = 1 + left_internal + right_internal

    # Add leaf nodes from left and right subtrees
    leaf_nodes = left_leaves + right_leaves

    return internal_nodes, leaf_nodes

def draw_decision_boundary(model_function:Callable, grid_abs_bound:float=1.0,savefile:str=None):
    """`model_function` should be your model's formula for evaluating your decision tree, returning either `0` or `1`.
    \n`grid_abs_bound` represents the generated grids absolute value over the x-axis, default value generates 50 x 50 grid.
    \nUse `grid_abs_bound = 1.0` for question 6 and `grid_abs_bound = 1.5` for question 7.
    \nSet `savefile = 'plot-save-name.png'` to save the resulting plot, adjust colors and scale as needed."""



    colors=['#91678f','#afd6d2'] # hex color for [y=0, y=1]



    xval = np.linspace(grid_abs_bound,-grid_abs_bound,50).tolist() # grid generation
    xdata = []
    for i in range(len(xval)):
        for j in range(len(xval)):
            xdata.append([xval[i],xval[j]])



    df = pd.DataFrame(data=xdata,columns=['x_1','x_2']) # creates a dataframe to standardize labels
    df['y'] = df.apply(model_function,axis=1) # applies model from model_function arg
    d_columns = df.columns.to_list() # grabs column headers
    y_label = d_columns[-1] # uses last header as label
    d_xfeature = d_columns[0] # uses first header as x_1 feature
    d_yfeature = d_columns[1] # uses second header as x_1 feature
    df = df.sort_values(by=y_label) # sorts by label to ensure correct ordering in plotting loop



    d_xlabel = f"feature  $\mathit{{{d_xfeature}}}$" # label for x-axis
    dy_ylabel = f"feature  $\mathit{{{d_yfeature}}}$" # label for y-axis
    plt.figure(figsize=(5,5)) # set figure size (width,height
    plt.xlabel(d_xlabel, fontsize=10) # set x-axis label
    plt.ylabel(dy_ylabel, fontsize=10) # set y-axis label
    legend_labels = [] # create container for legend labels to ensure correct ordering



    for i,label in enumerate(df[y_label].unique().tolist()): # loop through placeholder dataframe
        df_set = df[df[y_label]==label] # sort according to label
        set_x = df_set[d_xfeature] # grab x_1 feature set
        set_y = df_set[d_yfeature] # grab x_2 feature set
        plt.scatter(set_x,set_y,c=colors[i],marker='s', s=40) # marker='s' for square, s=40 for size of squares large enough
        legend_labels.append(f"""{y_label} = {label}""") # apply labels for legend in the same order as sorted dataframe



    plt.title("Model Decision Boundary", fontsize=12) # set plot title
    ax = plt.gca() # grab to set background color of plot
    ax.set_facecolor('#2b2d2e') # set aforementioned background color in hex color
    plt.legend(legend_labels) # create legend with sorted labels



    if savefile is not None: # save your plot as .png file
        plt.savefig(savefile)
    plt.show() # show plot with decision bounds

def model_y(row):
    prediction = predict(tree,row)
    return prediction

def calculate_n_errn(xtrain,xtest,ytrain,ytest):
    tree = decision_tree_classifier(xtrain, ytrain)
    internal_count, leaf_count = count_nodes(tree)
    node_num = internal_count + leaf_count
    print("Number of nodes: ",node_num)
    predictlabel = np.zeros(xtest.shape[0], dtype=int)
    for i in range(xtest.shape[0]):
        X_new = xtest[i, :]  # Get the i-th data point
        predictlabel[i] = predict(tree, X_new)
    predict_error = np.sum(predictlabel != ytest) / len(ytest)
    print("Prediction error: %.3f" % predict_error)
    
    return node_num,predict_error

# Example usage:
if __name__ == "__main__":
    # # Assuming you have loaded your data into X_train and y_train
    # feature, instances = readfiles('./hw02/data/D2.txt')
    # predicted_labels = np.zeros(feature.shape[0], dtype=int)
    # feature2, instances2 = readfiles('./hw02/data/D2.txt')
    # # Build the decision tree
    # tree = decision_tree_classifier(feature, instances)
    # #print("Number of nodes: ",node_num)
    # for i in range(feature2.shape[0]):
    #     X_new = feature2[i, :]  # Get the i-th data point
    #     predicted_labels[i] = predict(tree, X_new)
    # #print(predicted_labels)
    # predict_error = np.sum(predicted_labels != instances) / len(instances)
    # print("Prediction error: %.3f" % predict_error)
    # # Call the count_nodes function to get counts
    # internal_count, leaf_count = count_nodes(tree)
    # # print("Number of internal nodes: %d" % internal_count)
    # # print("Number of leaf nodes: %d" % leaf_count)
    # print("Number of nodes: %d" % (internal_count + leaf_count))
    
    

    # # Assuming you have a trained decision tree named 'tree'
    # # You should replace feature_names and class_names with your actual feature names and class labels
    # dot = graphviz.Digraph(format='png')
    # tree_to_dot(tree, dot, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'])

    # # Save the decision tree visualization as a PNG or display it
    # #dot.render("./hw02/data/custom_decision_tree_D2")  # Saves as custom_decision_tree.png by default
    # dot.view("custom_decision_tree")    # Opens the custom_decision_tree.png file in the default viewer
    
    # draw_decision_boundary(model_function=model_y, grid_abs_bound=1)
    dataset = np.loadtxt("./hw02/data/Dbig.txt", delimiter=' ')
    np.random.shuffle(dataset)
    data = dataset[0:8192,:]
    X_test = dataset[8192:,0:2]
    y_test = dataset[8192:,2]
    training_size = [32, 128, 512, 2048, 8192]
    n_values = []
    errn_values = []
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

        tree = decision_tree_classifier(X_train, y_train)
        draw_decision_boundary(model_function=model_y, grid_abs_bound=1.5)
        # Calculate n and errn for this dataset
        #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        n, errn = calculate_n_errn(X_train, X_test, y_train, y_test)
        n_values.append(n)
        errn_values.append(errn)
    
    plt.figure()
    plt.plot(n_values, errn_values, 'o-',label='error versus number of nodes')
    plt.xlabel('number of nodes')
    plt.ylabel('error rate')
    plt.legend()
    plt.show()