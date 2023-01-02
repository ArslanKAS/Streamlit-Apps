import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from plotly import express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Make containers to divide our dashboard into parts
st.set_page_config(layout="wide")
header = st.container()
data_sets = st.container()
sidebar = st.sidebar
features = st.container()
model_training = st.container()

SEED = 40

############################# HEADER SECTION #################################
with header:
    st.title("Titanic Price Prediction App")
    st.text("In this experimental project we will work with Titanic dataset")
    st.image('Titanic_Prediction.png')
    st.write("Name: Arsalan Ali")
    st.write("Email: arslanchaos@gmail.com")

########################## DATAFRAME AND PLOTS #################################
with data_sets:
    st.header("The Dataset")
    st.text("Data of all people who survived or not")
    # Import dataset
    df = sns.load_dataset("titanic")
    df = df.dropna()


    # Show dataset
    st.dataframe(df.head(10))

    # making columns
    left_column, right_column = st.columns(2)

    with left_column:
        # Num of Passengers
        st.subheader("Passengers according to sex")
        count_of_passengers = df['sex'].value_counts()
        st.text(count_of_passengers)
        st.bar_chart(count_of_passengers)

    with right_column:
        # barplot different
        st.subheader("Passengers according to age")
        count_of_passengers = df[df["sex"].isin(["age"])].value_counts()
        st.text(count_of_passengers)
        st.bar_chart(df["age"].sample(10))

    # other plot
    st.subheader("Passengers according to class")
    st.bar_chart(df["class"].value_counts())

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df2 = df
    df2_name = df2.columns
############################# CORRELATION #################################

    # Temporarily Feature Encoding in order to show Correlation
    for col in df2_name:
        if df2[col].dtypes != "int64" and df2[col].dtypes != "float64":
            df2[col] = le.fit_transform(df2[col])


    df2 = pd.DataFrame(df2, columns = df2_name)

    # Correlation Heatmap Function
    def correlation_heatmap(df):
        _ , ax = plt.subplots(figsize =(14, 12))
        colormap = sns.diverging_palette(220, 10, as_cmap = True)
        
        _ = sns.heatmap(
            df.corr(), 
            cmap = colormap,
            square=True, 
            cbar_kws={'shrink':.9 }, 
            ax=ax,
            annot=True, 
            linewidths=0.1,vmax=1.0, linecolor='white',
            annot_kws={'fontsize':12 })
        plt.title('Correlation of Features', y=1.05, size=15)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    fig = correlation_heatmap(df2)
    st.pyplot(fig,0)

############################# SIDEBAR MENU #################################
with sidebar:

    st.header("HYPER-PARAMETERS")
    st.write("-------------------")

    # Select the sample size for Dataset
    st.subheader("Sample Size")
    dataframe_size = st.slider("How many passengers?", min_value=5, max_value=150, value=20, step=5)

    # Select the Depth of Forest
    st.subheader("Depth of Model")
    max_depth = st.slider("How dense the model should be?", min_value=10, max_value=100, value=20, step=5)
    df = df.sample(dataframe_size)

    # n_estimators
    st.subheader("Number of Trees")
    n_estimators = st.selectbox("How many trees should be in RF?", options=[5,10,15,20,50,100,"No limit"])

############################# FEATURE SELECTION #################################
with model_training:
    st.header("Feature Selection")
    st.text("Select the features you want to include in model training")

    # making columns
    user_input, display = st.columns(2)


    # USER INPUT COLUMN
    with user_input:

        # input featues from user
        features_input = df.drop("fare", axis=1).columns
        input_features = user_input.multiselect('Select the features',
                                    features_input.tolist(),
                                        ["age", "parch"])

############################# MODEL SELECTION #################################
    # Selecting the Model
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    # If no parameters given then use defaults
    if n_estimators == "No limit":
        model = RandomForestRegressor(max_depth=max_depth)

    # Split the data into Train and Test
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.25, random_state=0)

    # Splitting before doing any Feature Engineering to avoid Data Leakage
    X_train, y_train = train[input_features], train[['fare']].values
    X_test, y_test = test[input_features], test[['fare']].values

################ FEATURE ENCODING AND FEATURE SCALING ###########################
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler

    le = LabelEncoder()
    trans = MinMaxScaler()
    X_train_name = X_train.columns

    # Feature Encoding
    for col in X_train_name:
        if X_train[col].dtypes != "int64" and X_train[col].dtypes != "float64":
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.fit_transform(X_test[col])

    # Feature Scaling
    X_train = trans.fit_transform(X_train)
    X_test = trans.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns = X_train_name)
    X_test = pd.DataFrame(X_test, columns = X_train_name)

############################# FEATURE IMPORTANCE #################################

with features:

    st.header("Feature Importance:")
    st.text("Lets see which features are important")

    from sklearn.ensemble import ExtraTreesRegressor

    imp = ExtraTreesRegressor(n_estimators=350, random_state=SEED)

    imp.fit(df2.drop("fare", axis=1) ,df2["fare"])

    # Plot feature importance
    feature_importance = imp.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    fig = px.bar(y=feature_importance, x=df2_name[sorted_idx].tolist(), text_auto='.2s',
            title="Default: various text sizes, positions and angles")

    fig.update_layout(
    xaxis_title="Important Features",
    yaxis_title="Value")

    st.write(fig)

############################# MODEL TRAINING #################################

    # Fit our model
    model.fit(X_train, np.ravel(y_train))

    # Predict
    pred = model.predict(X_test)

############################# MODEL EVALUATION #################################
    with display:
        st.markdown(" **R2 score:** ")
        st.write("The probability of model to predict correctly")
        st.write(round(r2_score(y_test, pred),2))

        st.markdown(" **Mean squared error(MSE):** ")
        st.write("The squared difference between predicted and actual value")
        st.write(round(mean_squared_error(y_test, pred),2))

        st.markdown(" **Root mean squared error(RMSE):** ")
        st.write("The difference between predicted and actual value")
        st.write(round(np.sqrt(mean_squared_error(y_test, pred)),2))

        st.markdown(" **Mean absoulte error(MAE):** ")
        st.write("The difference (without direction) between predicted and actual value")
        st.write(round(mean_absolute_error(y_test, pred),2))


