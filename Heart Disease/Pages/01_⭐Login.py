import streamlit as st
import sqlite3
st.set_page_config(layout="wide")

# Create a connection to the database
conn = sqlite3.connect('Database/project_database.db')
c = conn.cursor()


# Define the login page
def login_page():
    st.title("Please Enter Your Login Details")
    st.image("Images/Login.png", width=400)

    # Get the username/email and password from the user
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Try to find the user in the database
    c.execute("SELECT * FROM users WHERE (username=? OR email=?) AND password=?", (username, username, password))
    user = c.fetchone()
    try:
        st.session_state["user"] = user[2]
    except:
        pass

    # If the user is found, show a success message and allow them to access their account
    if user is not None:
        st.success("Logged in as {}".format(user[1]))
        # Do whatever you want to do with the user's account here
    # Otherwise, show an error message
    else:
        st.error("Invalid username/email or password")

# Run the app
if __name__ == "__main__":
    login_page()