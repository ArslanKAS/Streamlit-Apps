import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(layout="wide")

########################## HEADER ############################

st.image('COVID19-app.png')
# Reading csv data
df = pd.read_csv('Covid-19.csv')
# Displaying Data and its Shape
st.subheader("COVID-19 Dataset", df)
st.write("Shape of data", df.shape)

############################ SIDEBAR ###########################

with st.sidebar:
    st.header("User Input")
    # Creating selectbox for Graphs & Plots
    graphs = st.selectbox("Graphs & Plots", ("Bar Graph", "Scatter Plot", "HeatMap", "Pie Chart"))
    # Sorting the columns
    index = sorted(df.columns.unique())
    # Setting default value for x, y, and color
    default_index_x = index.index('State/UTs')
    default_index_y = index.index('Total Cases')
    default_index_col = index.index('Death Ratio (%)')


    # Creating selectbox for x, y and color label and setting default value
    x_label = st.selectbox("X label Parameter", index, index=default_index_x)
    y_label = st.selectbox("Y label Parameter", index, index=default_index_y)
    col = st.selectbox("Color", index, index=default_index_col)

############################ VISUALIZATION ###########################

# function to plot graphs
def visualize_plotly(graph):
    if graph == "Bar Graph":
        st.write(graph)
        fig = px.bar(df, x=x_label, y=y_label, color=col)

    elif graph == "Scatter Plot":
        st.write(graph)
        fig = px.scatter(df, x=x_label, y=y_label, color=col)

    elif graph == "HeatMap":
        st.write(graph)
        fig = px.density_heatmap(df, x=x_label, y=y_label, nbinsx=20, nbinsy=20)

    else:
        st.write(graph)
        fig = px.pie(df, values=x_label, names=df[y_label])

    return fig

figure = visualize_plotly(graphs)

st.plotly_chart(figure)

############################ REPORTING ###########################

# Creating buttons to display reports
if st.button("Highest Cases"):
    st.header("Highest Cases in a State/UT")
    highest_cases = df[df['Total Cases'] == max(df['Total Cases'])]
    st.write(highest_cases)

if st.button("Lowest Cases"):
    st.header("Lowest Cases in a State/UT")
    lowest_cases = df[df['Total Cases'] == min(df['Total Cases'])]
    st.write(lowest_cases)

if st.button("Highest Active Cases"):
    st.header("Highest Active Cases in a State/UT")
    high_active_cases = df[df['Active'] == max(df['Active'])]
    st.write(high_active_cases)

if st.button("Lowest Active Cases"):
    st.header("Lowest Active Cases in a State/UT")
    low_active_cases = df[df['Total Cases'] == min(df['Total Cases'])]
    st.write(low_active_cases)

if st.button("Highest Death Ratio (%)"):
    st.header("Highest Death Ratio (%) in a State/UT")
    high_death = df[df['Death Ratio (%)'] == max(df['Death Ratio (%)'])]
    st.write(high_death)

if st.button("Lowest Death Ratio (%)"):
    st.header("Lowest Death Ratio (%) in a State/UT")
    low_death = df[df['Death Ratio (%)'] == min(df['Death Ratio (%)'])]
    st.write(low_death)



    





