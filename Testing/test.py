import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

st.write('''
#  Explore different models and datasets
''')
dataset_name=st.sidebar.selectbox(
    'Select Dataset',
    ('Iris','Breast Cancer','Wine')
)
classifier_name=st.sidebar.selectbox(
    'Select Classifier',
    ('KNN','SVM','Random Forest')
)
def get_dataset(dataset_name):
    data=None
    if dataset_name=='Iris':
        data=datasets.load_iris()
    elif dataset_name=='wine':
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target
    return x,y
    
X, y =get_dataset(dataset_name)
st.write('Shape ',X.shape)
st.write('Classes ',len(np.unique(y)))
def add_parameter_ui(classifier_name):
    params=dict()
    if classifier_name=='SVM':
        C=st.sidebar.slider('C',0.01,10.0)
        params['C']=C
    elif classifier_name=='KNN':
        K=st.sidebar.slider('K',1,15)
        params['K']=K
    else:
        max_depth=st.sidebar.slider('max_depth',2,15)
        params['max_depth']=max_depth
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators
    return params
params=add_parameter_ui(classifier_name)
def get_classifier(classifier_name,params):
    clf=None
    if classifier_name=='SVM':
        clf=SVC(C=params['C'])
    elif classifier_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],random_state=1234)
    return clf
clf=get_classifier(classifier_name,params)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)
st.write(f'Classifier={classifier_name}')
st.write(f'Accuracy={acc}')
pca=PCA(2)
X_projected=pca.fit_transform(X)
x1=X_projected[:,0]
x2=X_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
# plt.show()
st.pyplot(fig)
# st.header('Hello World')
# st.text('hello world!')
# df=sns.load_dataset('iris')
# st.write(df.head())
# st.bar_chart(df['sepal_length'])
# st.line_chart(df['petal_length'])
# x = st.slider('x')  
# st.write(x, 'squared is', x * x)
# df=sns.load_dataset('titanic')
# st.write(df.head())
# st.bar_chart(df['sex'].value_counts())
# st.subheader('Class Difference')
# st.bar_chart(df['class'].value_counts())
# st.bar_chart(df['age'].sample())
# header=st.container()
# data_sets=st.container()
# features=st.container()
# st.markdown('Features')
# model_training=st.container()
# with header:
#     st.title('Titanic App')
#     st.text('Data')
# with data_sets:
#     st.header('Titanic Destroyed')
# with features:
#     st.header('Features')
# with model_training:
#     st.header('Model Training')
# input,display=st.columns(2)
# input.slider('How Much',min_value=10,max_value=100,value=20,step=5)