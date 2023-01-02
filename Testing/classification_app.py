# Name: Arsalan Ali
# Email: arslanchaos@gmail.com

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets



from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA

# Setting the title of the Webpage
st.title("Classification App (KAS)")

# Setting the display box for Dataset on Left Sidebar
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))

# Setting the display of ML Model on Left Sidebar
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

# Setting the title of the Webpage
st.write(f"## {dataset_name} prediction using {classifier_name}")

# Function to select the Dataset chosen by User
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

# Initializing the function to get the dataset
# X = features and Y = target
X, y = get_dataset(dataset_name)

# Showing the Shape of X
st.write("shape of dataset", X.shape)

# Showing number of unique values/classes in Y
st.write("number of classes", len(np.unique(y)))

# Function to set Parameters of the ML Models
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        # K is the number of neighbours
        K = st.sidebar.slider("K", 1, 15)
        # Adding K key and value to dictionary params
        params["K"] = K
    elif clf_name == "SVM":
        # C is used to control the error during classification
        C = st.sidebar.slider("C", 0.01, 10.0)
        # Adding C key and value to dictionary params
        params["C"] = C
    else:
        # Max depth is the depth of the trees (nodes within nodes)
        max_depth = st.sidebar.slider("max_depth", 2, 20)
        # N_estimator is the number of trees
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        # Adding Max depth key and value to dictionary params
        params["max_depth"] = max_depth
        # Adding N_estimator key and value to dictionary params
        params["n_estimators"] = n_estimators
    return params

# Initializing the parameter function
params = add_parameter_ui(classifier_name)

# Function to select the ML Model
def get_classifer(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)
    return clf

# Initializing the ML Model function
clf = get_classifer(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

# Plotting 
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="rocket_r")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)

# Plotting Confusion Matrix
fig2 = plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="viridis")
plt.ylabel("Actual Output")
plt.xlabel("Predicted Output")
all_sample_title = "Accuracy Score: {0}".format(acc)
plt.title(all_sample_title, size = 8)

st.pyplot(fig2)
