import streamlit as st
import numpy as np
import pandas as pd
import cufflinks as cf

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

@st.cache
def get_data():
    df = pd.read_csv("owid-covid-data.csv", nrows=200)
    df["date"] = pd.to_datetime(df.date).dt.date
    df['date'] = pd.DatetimeIndex(df.date)

    return df

url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
data = get_data()

st.title("Web App by Alina Ayub Mayo")
daily_cases = data.groupby(pd.Grouper(key="date", freq="1D")).aggregate(new_cases=("new_cases", "sum")).reset_index()
fig = daily_cases.iplot(kind="line", asFigure=True, 
                        x="date", y="new_cases")
st.plotly_chart(fig)

# url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
# data = get_data(url)

locations = data.location.unique().tolist()

sidebar = st.sidebar
location_selector = sidebar.selectbox(
    "Select a Location",
    locations
)
st.markdown(f"# Currently Selected {location_selector}")

daily_cases = data.groupby(pd.Grouper(key="date", freq="1D")).aggregate(new_cases=("new_cases", "sum")).reset_index()
fig = daily_cases.iplot(kind="line", asFigure=True, 
                        x="date", y="new_cases")
st.plotly_chart(fig)

show_data = sidebar.checkbox("Show Data")

if show_data:
    st.dataframe(data)

daily_cases = data.groupby(pd.Grouper(key="date", freq="1D")).aggregate(new_cases=("new_cases", "sum")).reset_index()
fig = daily_cases.iplot(kind="line", asFigure=True, 
                        x="date", y="new_cases")
st.plotly_chart(fig)