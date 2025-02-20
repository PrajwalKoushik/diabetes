import pickle
import streamlit as st

# Set up the page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

# Load the trained diabetes model
diabetes_model_path = r"C:\Users\prajw\OneDrive\Desktop\workshop2\diabetes_model.sav"
try:
    with open(diabetes_model_path, 'rb') as file:
        diabetes_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please check the path and try again.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Title of the application
st.title('Diabetes Prediction Using Machine Learning')

# Create input fields for user data
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)

with col2:
    Glucose = st.number_input('Glucose Level', min_value=0)

with col3:
    BloodPressure = st.number_input('Blood Pressure Value', min_value=0)

with col1:
    SkinThickness = st.number_input('Skin Thickness Value', min_value=0)

with col2:
    Insulin = st.number_input('Insulin Level', min_value=0.0, format="%.2f")

with col3:
    BMI = st.number_input('BMI Value', min_value=0.0, format="%.2f")

with col1:
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function Value', min_value=0.0, format="%.3f")

with col2:
    Age = st.number_input('Age of the Person', min_value=0, step=1)

# Initialize the diagnosis variable
diab_diagnosis = ''

# Button for prediction
if st.button('Diabetes Test Result'):
    try:
        # Prepare the input data for prediction
        user_input = [
            float(Pregnancies),
            float(Glucose),
            float(BloodPressure),
            float(SkinThickness),
            float(Insulin),
            float(BMI),
            float(DiabetesPedigreeFunction),
            float(Age)
        ]

        # Perform the prediction
        diab_prediction = diabetes_model.predict([user_input])

        # Interpret the prediction result
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic.'
        else:
            diab_diagnosis = 'The person is not diabetic.'

        # Display the result
        st.success(diab_diagnosis)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
if st.button('Show Model Accuracy'):

    test_data = pd.read_csv(r"C:\Users\prajw\OneDrive\Desktop\workshop2\diabetes.csv")

    x_test = test_data.drop(columns=['Outcome'])
    y_test = test_data['Outcome']

    y_pred = diabetes_model.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)

    st.write(f"Model Accuracy on Test Data:{accuracy * 100:.2f}%")


