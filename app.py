pip install scikit-learn
import streamlit as st
import pandas as pd

# Attempt to import scikit-learn and RandomForestClassifier
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    sklearn_available = True
except ImportError:
    sklearn_available = False

# Check if Matplotlib is available
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

if not sklearn_available:
    st.error("scikit-learn is not available. Please make sure it is installed in your environment.")
    st.stop()

# Load the diabetes dataset
df = pd.read_csv('diabetes.csv')

# Sidebar inputs for user data
st.sidebar.title('Diabetes Prediction')
st.sidebar.header('Enter Patient Data')
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, step=1, value=0)
glucose = st.sidebar.slider('Glucose', min_value=0, max_value=200, step=1, value=100)
blood_pressure = st.sidebar.slider('Blood Pressure', min_value=0, max_value=122, step=1, value=70)
skin_thickness = st.sidebar.slider('Skin Thickness', min_value=0, max_value=100, step=1, value=20)
insulin = st.sidebar.slider('Insulin', min_value=0, max_value=846, step=1, value=79)
bmi = st.sidebar.slider('BMI', min_value=0.0, max_value=67.1, step=0.1, value=25.0)
age = st.sidebar.slider('Age', min_value=21, max_value=90, step=1, value=30)

# Create a DataFrame with user input data
user_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'Age': [age]
})

# Load preprocessed data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the Random Forest model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Make predictions
prediction = rf.predict(user_data)

# Display prediction result
st.title('Diabetes Prediction Result')
st.header('Patient Data:')
st.write(user_data)
st.header('Prediction:')
if prediction[0] == 0:
    st.write('No diabetes detected.')
else:
    st.write('Diabetes detected.')

# Display model accuracy
st.header('Model Accuracy:')
accuracy = accuracy_score(y_test, rf.predict(x_test))
st.write(f'Model Accuracy: {accuracy * 100:.2f}%')

# Optional: Display additional information or recommendations based on prediction result
if prediction[0] == 1:
    st.header('Recommendations:')
    st.write('Based on the prediction result, it is advisable to consult a healthcare professional for further evaluation and guidance.')

# Display a message if Matplotlib is not available
if not matplotlib_available:
    st.error('Matplotlib is not available. Please make sure it is installed in your environment.')
