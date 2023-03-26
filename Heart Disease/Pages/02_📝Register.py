import streamlit  as st
import sqlite3
st.set_page_config(layout="wide")

# Create a connection to the database
conn = sqlite3.connect('Database/project_database.db')
c = conn.cursor()

# Create a table to store user registration details if it doesn't already exist
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT, 
             full_name TEXT NOT NULL, 
             username TEXT NOT NULL UNIQUE, 
             password TEXT NOT NULL, 
             age INTEGER NOT NULL, 
             gender TEXT NOT NULL, 
             email TEXT NOT NULL UNIQUE, 
             city TEXT NOT NULL, 
             phone_number TEXT NOT NULL UNIQUE, 
             disease INTEGER NOT NULL)''')
conn.commit()

# Define the registration page
def register_page():
    st.title("Register for Disease Detection")

    # Get the user's registration details

    col1, col2, col3 = st.columns((2,1,1))
    full_name = col2.text_input("Full Name", placeholder="John Doe")
    col1.image("Images/Register.png", width=400)
    username = col2.text_input("Username", placeholder="imjohn123")
    password = col3.text_input("Password", type="password")
    age = col2.number_input("Age", min_value=0, max_value=150)
    gender = col3.selectbox("Gender", options=["Male", "Female", "Other"])
    email = col2.text_input("Email", placeholder="johndoe@hotmail.com")
    city = col3.text_input("City", placeholder="New York")
    phone_number = col3.text_input("Phone Number", placeholder="03331234567")


    # If the user clicks the "Register" button, insert their details into the database
    if col2.button("Submit"):
        if email.strip() == '' or city.strip() == '' or phone_number.strip() == '':
            st.error("Please enter values for all fields")
        else:
            c.execute("INSERT INTO users (full_name, username, password, age, gender, email, city, phone_number, disease) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (full_name, username, password, age, gender, email, city, phone_number, 0))
            conn.commit()
            conn.close()
            st.success("Registration successful. Please log in.")

# Run the app
if __name__ == "__main__":
    register_page()