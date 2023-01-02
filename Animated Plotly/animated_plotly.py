import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px 
st.set_page_config(layout="wide")

####################### HEADER ######################

st.image('animated_plotly.png')
# Import dataset
df = px.data.gapminder()
st.subheader("The Dataset")
st.write(df)

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("List of Features")
    st.write(df.columns)

with right_col:
    # Descriptive Statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

# Data Management
with st.sidebar:
    year_option = df["year"].unique().tolist()
    st.header("Sidebar")
    year = st.selectbox("select the year to plot:", year_option, 0)

##################### PLOTTING #########################
st.subheader("The Graph")
fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="country", hover_name="continent",
                log_x=True, size_max=55, range_x=[100,100000], range_y=[20,90],
                animation_frame="year", animation_group="country")

fig.update_layout(width=800, height=600)

st.write(fig)
