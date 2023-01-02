import streamlit as st
from plotly import express as px
import pandas as pd
from PIL import Image
st.set_page_config(layout="wide")


############################# HEADER SECTION #################################
image = Image.open('UFO_banner.png')

st.title("UFO Sightings App")
st.image(image, caption='Are we alone?')
st.write("Name: Arsalan Ali")
st.write("Email: arslanchaos@gmail.com")

# import dataset
# df = px.data.stocks()
df = pd.read_csv("UFO.csv", nrows=200)
df = df.dropna()
st.header(f"Dataset: UFO sightings")
st.write(df.head())

########################## DESCRIPTIVE STATISTICS ############################

st.header("Basic Statistics")
st.write(df.describe())

############################# SIDEBAR #################################
with st.sidebar:
    st.subheader("List of Columns")
    st.write(df.columns)

    # data management
    shape_options = df["shape"].unique().tolist()

    shape = st.selectbox("Select shape to plot:" , shape_options)

    df = df[df["shape"]== shape]

############################# PLOTS #################################

fig = px.scatter_polar(df, r="city", theta="duration (hours/min)")
# fig = px.scatter(df, x="latitude", y="latitude", size="duration (seconds)", color="country",hover_name="city")
 #log_x=True, size_max=55, range_x=[100,10000], range_y=[20,90]
fig.update_layout(width=800,height=600)

st.header("UFO sightings based on shape")
st.subheader(f"The shape: {shape}")
st.write(fig)