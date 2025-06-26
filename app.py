import streamlit as st
import json
import numpy as np
import joblib 

# Load the columns from the JSON file
try:
    with open('columns.json', 'r') as f:
        data_columns = json.load(f)['data_columns']
except FileNotFoundError:
    st.error("Error: columns.json not found. Make sure it's in the same directory as the app.")
    data_columns = [] 

# The first three columns are features, the rest are locations
locations = data_columns[3:]

# Load the trained model (replace 'bangalore_house_price_model.joblib' with your model file)
# It's crucial that your model expects the input features in the same order as data_columns
try:
    with open('banglore_house_price_prediction', 'rb') as f:
        model = joblib.load(f)
except FileNotFoundError:
    st.error("Error: bangalore_house_price_prediction.joblib not found. Please place your trained model file in the same directory.")
    model = None # Indicate that model is not loaded


# --- Prediction Function ---
def predict_price(location, sqft, bath, bhk):
    """
    Predicts the house price based on the input features.
    You need to integrate your actual model prediction logic here.
    """
    if model is None:
        return "Model not loaded. Cannot predict."

    try:
        # Create a zero array with the same length as data_columns
        x = np.zeros(len(data_columns))

        # Set the first three features
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        # Find the index of the selected location
        try:
            loc_index = data_columns.index(location)
        except ValueError:
            # Handle cases where the location might not be found (e.g., "other")
            # In your case, 'other' was dropped, so this should only happen if
            # an unlisted location somehow gets selected.
            st.warning(f"Location '{location}' not found in model columns. Using default (first location).")
            loc_index = 3 # Default to the first location if not found

        if loc_index >= 0:
            x[loc_index] = 1 # Set the dummy variable for the selected location to 1

        # Make the prediction
        price = model.predict([x])[0] # Assuming model.predict takes a list of arrays and returns an array

        return round(price, 2) # Round to 2 decimal places
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Error during prediction."


# --- Streamlit UI ---
st.set_page_config(page_title="Bangalore House Price Predictor", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2e86de;
        text-align: center;
        margin-bottom: 30px;
    }
    .stSelectbox, .stNumberInput {
        border-radius: 8px;
        padding: 5px;
        border: 1px solid #ccc;
    }
    .stButton > button {
        background-color: #2e86de;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 20px;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1a5e9f;
    }
    .stMetric > div > div {
        color: #28a745;
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🏡 Bangalore House Price Predictor")

st.write("Enter the details below to get an estimated house price in Lakhs.")

# Input fields
total_sqft = st.number_input("Total Square Feet", min_value=500.0, max_value=10000.0, value=1000.0, step=100.0)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
bhk = st.number_input("BHK (Bedrooms, Hall, Kitchen)", min_value=1, max_value=10, value=2, step=1)
location = st.selectbox("Select Location", locations)

# Prediction button
if st.button("Predict Price"):
    if model is None:
        st.warning("Model is not loaded. Please ensure 'bangalore_house_price_model.joblib' is in the correct directory.")
    else:
        # Call the prediction function
        predicted_price = predict_price(location, total_sqft, bath, bhk)

        # Display the result
        if isinstance(predicted_price, (int, float)):
            st.metric(label="Estimated Price", value=f"{predicted_price} Lakhs")
            st.success("Prediction successful!")
        else:
            st.error(predicted_price) # Display error message from the predict_price function

st.markdown("---")
st.markdown("This app uses a machine learning model to estimate house prices in Bangalore.")
