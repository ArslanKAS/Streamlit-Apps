import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import sqlite3
from PIL import Image
st.set_page_config(layout="wide")

def my_Show():
    switch_page("predict")

# Define the main function that runs the app
def main():
    st.title("Welcome to Heart Disease Detection")
    st.image("Images/Welcome.png", width=500)

    st.subheader("Want To Try Out Our Prediction Model?")
    if st.button("Let's Predict"):
        my_Show()

    tab_1, tab_2 = st.tabs(["Developer", "Dataset Info"])
    tab_1.markdown("#### Project Developer")
    tab_1.markdown("**Developer:** Amna Riaz")
    tab_1.markdown("**Details:** As a final year student, I am interested in predicting the likelihood of heart disease, and the dataset provided by the UCI \
                   Machine Learning Repository seems to be a valuable resource for this purpose. This dataset includes various medical and demographic\
                   information for patients, such as age, sex, cholesterol level, and electrocardiogram readings, and has been collected from multiple institutions\
                   for research purposes in the field of cardiovascular health.\
                   By analyzing this data and using machine learning techniques, I have developed a model that can accurately predict the probability of heart disease \
                   in patients based on their individual characteristics.")

    tab_2.markdown("#### Dataset Creators")
    tab_2.markdown("1. Hungarian Institute of Cardiology. Budapest: **Andras Janosi, M.D.**")
    tab_2.markdown("2. University Hospital, Zurich, Switzerland: **William Steinbrunn, M.D.**")
    tab_2.markdown("3. University Hospital, Basel, Switzerland: **Matthias Pfisterer, M.D.**")
    tab_2.markdown("4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: **Robert Detrano, M.D., Ph.D.**")


# Run the app
if __name__ == "__main__":
    main()
