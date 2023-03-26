import sqlite3
import streamlit as st
st.set_page_config(layout="wide")


# Create a connection to the database
conn = sqlite3.connect('Database/project_database.db')
c = conn.cursor()

# Define the registration page
def account():
    st.title("User Account Details")
    col1, col2 = st.columns([2,2])
    col2.image("Images/Account.png", width=300)

    try:
        c.execute('SELECT * FROM users WHERE username=? ', (st.session_state["user"],))
        user = c.fetchone()
        column_names = ["ID","Full Name","Username", "Password", "Age", "Gender", "Email", "City", "Phone Number", "Disease Stage"]
        col1.table(zip(column_names,user))
    except:
        st.error("Please Login to View Account")

    tab1, tab2, tab3 = st.tabs(["Heart Disease Stage", "Age", "Gender"])
    tab1.markdown("#### Angiographic Disease Status")
    tab1.markdown("**Stage 0:** Angiographic Diameter Narrowing Less than 50%")
    tab1.markdown("**Stage 1-4:** Diameter Narrowing Greater than 50%")

    tab2.markdown("#### Role of Age")
    tab2.markdown("Age is a significant factor in determining the prevalence and severity of angiographic disease status, as the risk of \
                   developing atherosclerosis and other cardiovascular diseases increases with age. Older patients are more likely to have more \
                   extensive and severe coronary artery disease compared to younger patients.")
    
    tab3.markdown("#### Role of Gender")
    tab3.markdown("Gender also plays a role in angiographic disease status, as men tend to have a higher prevalence of coronary artery \
                   disease and atherosclerosis compared to premenopausal women. However, after menopause, the risk of cardiovascular disease in \
                  women increases, and the gender gap narrows.")


# Run the app
if __name__ == "__main__":
    account()