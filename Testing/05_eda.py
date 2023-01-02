import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Header Image
st.image("EDA_app_cover.png", caption='Arsalan Ali | arslanchaos@gmail.com')

# Title of Web App
st.markdown("""
# **Exploratory Data Analysis Web App**
This app is developed by Codanics Youtbue Channel called **EDA App**
""")

# Using Sidebar like this so other stuff can come in Sidebar
with st.sidebar:
    # Upload file from PC
    st.header("Upload your dataset (CSV)")
    uploaded_file = st.file_uploader("Upload your file", type=["csv"])

    # Dataset with URL for testing purpose
    URL = "https://raw.githubusercontent.com/emorisse/FBI-Hate-Crime-Statistics/master/2013/table13.csv"
    st.markdown(f"[Example CSV file]({URL})")

    # Dancing pandas
    st.image("https://i.pinimg.com/originals/25/17/39/251739c54e923b2bcc3c89252b6c0e56.gif", caption='dancing panda')
    # st.image("https://i.gifer.com/origin/22/22f37c8f7601e0f4847e99442433c5c4_w200.gif", caption='dancing panda')

# Pandas profiling Function and st.cache to improve speed
@st.cache(suppress_st_warning=True) 
def load_csv():
    pr = ProfileReport(df, explorative=True)
    st.header("**Input Dataframe**")
    st.write(df)
    st.write("---")
    st.header("**Profiling report with Pandas**")
    st_profile_report(pr)

# Conditions to Profile real dataset or the example through a button
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    load_csv()
else:
    st.info("Awaiting CSV file")
    if st.button("Press to use example data"):
        df = sns.load_dataset("tips")
        load_csv()
