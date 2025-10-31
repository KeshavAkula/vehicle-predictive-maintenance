import streamlit as st
import pickle
import numpy as np

# Load the trained model (replace 'model.pkl' with your actual file name)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the customized ranges for each feature based on dataset statistics
custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
    'Temperature_difference': (-22.669427, 119.008526)
}

# Feature descriptions
feature_descriptions = {
    'Engine rpm': 'Revolution per minute of the engine.',
    'Lub oil pressure': 'Pressure of the lubricating oil.',
    'Fuel pressure': 'Pressure of the fuel.',
    'Coolant pressure': 'Pressure of the coolant.',
    'lub oil temp': 'Temperature of the lubricating oil.',
    'Coolant temp': 'Temperature of the coolant.',
    'Temperature_difference': 'Temperature difference between components.'
}


# Function to predict engine condition
def predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp):
    input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure,
                           coolant_pressure, lub_oil_temp, coolant_temp]).reshape(1, -1)

    prediction = model.predict(input_data)[0]  # Regression output

    # Interpret the prediction value into categories
    if prediction < 0.5:
        condition = "Good"
        message = "The engine is in good condition."
    elif prediction < 1.5:
        condition = "Needs Maintenance"
        message = "The engine may need maintenance soon."
    else:
        condition = "Critical"
        message = "Warning! Engine condition is critical."

    return condition, message, prediction


# Streamlit App
def main():
    st.title("🚗 Engine Condition Prediction Dashboard")

    # Sidebar info
    st.sidebar.title("Feature Descriptions")
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}:** {description}")

    # Input sliders
    engine_rpm = st.slider("Engine RPM", *custom_ranges['Engine rpm'])
    lub_oil_pressure = st.slider("Lub Oil Pressure", *custom_ranges['Lub oil pressure'])
    fuel_pressure = st.slider("Fuel Pressure", *custom_ranges['Fuel pressure'])
    coolant_pressure = st.slider("Coolant Pressure", *custom_ranges['Coolant pressure'])
    lub_oil_temp = st.slider("Lub Oil Temperature", *custom_ranges['lub oil temp'])
    coolant_temp = st.slider("Coolant Temperature", *custom_ranges['Coolant temp'])
    temp_difference = st.slider("Temperature Difference", *custom_ranges['Temperature_difference'])

    # Predict button
    if st.button("🔍 Predict Engine Condition"):
        condition, message, raw_output = predict_condition(
            engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp)

        st.subheader(f"Predicted Condition: **{condition}**")
        st.write(message)
        st.write(f"Raw Model Output: {raw_output:.3f}")

    # Reset button
    if st.button("🔄 Reset Values"):
        st.rerun()


if __name__ == "__main__":
    main()
