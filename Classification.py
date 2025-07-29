import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Data
X = np.array([
    [22, 0], [25, 0], [30, 0.5], [20, 1],
    [18, 4], [17, 5], [20, 3], [19, 4.5],
    [35, 0], [32, 0.4], [16, 6], [21, 3.5]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])

# Model loading and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Student vs Professional Classifier", page_icon="🧠")

st.title("Student or Working Professional?")
st.write("Enter your details below to predict:")

age = st.number_input("Enter Age", min_value=10, max_value=100, step=1)
hours = st.slider("Hours of Study per Day", 0.0, 10.0, 1.0, step=0.5)

if st.button("Predict"):
    userdata = np.array([[age, hours]])
    predicted = model.predict(userdata)

    st.write(f"### Prediction Result:")
    if predicted[0] == 0:
        st.success("🧑‍💼 Likely a Working Professional.")
    else:
        st.success("🎓 Likely a Student.")
