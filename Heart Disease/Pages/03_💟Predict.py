import sqlite3
import joblib
import numpy as np
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier
st.set_page_config(layout="wide")


# Create a connection to the database
conn = sqlite3.connect('Database/project_database.db')
c = conn.cursor()

model = joblib.load("Model/heart_disease.joblib")
# Define the registration page
def predict():
    st.title("Enter Your Details")
    st.image("Images/Prediction.png", width=300)


    # Get the user's registration details
    chest_pain = {"Typical Angina":1, "Atypical Angina":2, "Non-anginal Pain":3,"Asymptomatic":4}
    Thalassemia = {"3: Normal":3, "6: Fixed":6, "7: Reversable":7}

    col1, col2, col3, col4 = st.columns(4)
    age = col1.number_input("Age", min_value=0, max_value=150)
    cp = col2.selectbox("Chest Pain", {"Typical Angina":1, "Atypical Angina":2, "Non-anginal Pain":3,"Asymptomatic":4})
    ca = col3.selectbox("Major Vessels",[0,1,2,3])
    chol = col4.number_input("Cholesterol (mg/dl)", min_value=10, max_value=350)
    thal = col1.selectbox("Thalassemia", {"3: Normal":3, "6: Fixed":6, "7: Reversable":7})
    thalach = col2.number_input("Heart Rate (bpm)", min_value=30, max_value=300)
    oldpeak = col3.number_input("ST Depression(mm)", min_value=0, max_value=10)

    if st.button("Predict"):
        # st.write(Thalassemia.get(thal))
        symptoms = (age,chest_pain.get(cp),ca,chol,Thalassemia.get(thal),thalach,oldpeak)
        # st.session_state["age"] = age
        results = model.predict([[*symptoms]])
        # st.session_state["results"] = results[0]
        if results[0] < 1:
            st.success("Angiographic Narrowing < 50%")
            st.markdown("**You seem normal but still it's best to consult a Cardiologist.**")
        else:
            st.error("Angiographic Narrowing > 50%")
            st.markdown("**You are not well. Visit a Cardiologist as soon as possible.**")

        try:
            
            # # Try to find the user in the database
            c.execute("SELECT * FROM users WHERE username=?", (st.session_state["user"],))
            user = c.fetchone()[2]
            # Update age and results in that specific user
            c.execute("""UPDATE users
                    SET disease = ?,
                        age = ?
                    WHERE
                        username = ?"""
                    , (results[0], age, user))
            
            conn.commit()
            conn.close()
        except:
            st.error("Please Login to Save Results")
# Run the app
if __name__ == "__main__":
    predict()