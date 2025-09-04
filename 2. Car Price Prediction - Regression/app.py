import joblib
import streamlit as st
import pandas as pd

# Function to extract categorical choices from the pipeline
def extract_categorical_choices_from_pipeline(pipeline):
    pre = pipeline.named_steps['preprocessor']  # ColumnTransformer
    choices = {}

    for name, trans, cols in pre.transformers_:
        if trans.__class__.__name__ == 'OneHotEncoder':
            ohe = trans
            for col_name, cats in zip(cols, ohe.categories_):
                choices[col_name] = list(cats)

    return choices

# Load the pipeline
pipeline = joblib.load('car_price_prediction_pipeline.pkl')

# Load the categorical choices
categorical_choices = extract_categorical_choices_from_pipeline(pipeline)

# Streamlit app
st.title(":red_car: Car Price Prediction")
st.write("Predict the price of a car based on its features.")
st.image("https://media.istockphoto.com/id/867003336/photo/rising-car-costs.jpg?s=612x612&w=0&k=20&c=1hyDp76LXFrysMjEXsSUlgr3629rDKCSFW1rXA-eveI=", use_container_width=True)


col1, col2 = st.columns(2)

# Numeric input features
with col1:
    st.subheader("Numeric Features")
    wheelbase = st.number_input("Wheelbase", min_value=0.0, value=95.0, step=0.1)
    carlength = st.number_input("Car Length", min_value=0.0, value=170.0, step=0.1)
    carwidth = st.number_input("Car Width", min_value=0.0, value=65.0, step=0.1)
    curbweight = st.number_input("Curb Weight", min_value=0.0, value=2500.0, step=1.0)
    cylindernumber = st.selectbox("Cylinder Number", options=[2, 3, 4, 5, 6, 8, 12], index=2)
    enginesize = st.number_input("Engine Size", min_value=0.0, value=130.0, step=0.1)
    boreratio = st.number_input("Bore Ratio", min_value=0.0, value=3.2, step=0.01)
    horsepower = st.number_input("Horsepower", min_value=0.0, value=100.0, step=1.0)
    citympg = st.number_input("City MPG", min_value=0.0, value=25.0, step=0.1)
    highwaympg = st.number_input("Highway MPG", min_value=0.0, value=30.0, step=0.1)

# Categorical input features
with col2:
    st.subheader("Categorical Features")
    brand = st.selectbox("Brand", options=categorical_choices['brand'])
    fueltype = st.selectbox("Fuel Type", options=categorical_choices['fueltype'])
    aspiration = st.selectbox("Aspiration", options=categorical_choices['aspiration'])
    carbody = st.selectbox("Car Body", options=categorical_choices['carbody'])
    drivewheel = st.selectbox("Drive Wheel", options=categorical_choices['drivewheel'])
    enginelocation = st.selectbox("Engine Location", options=categorical_choices['enginelocation'])
    enginetype = st.selectbox("Engine Type", options=categorical_choices['enginetype'])
    fuelsystem = st.selectbox("Fuel System", options=categorical_choices['fuelsystem'])

# Predict button
if st.button("Predict Price", type="primary", use_container_width=True):
    try:
        input_data = pd.DataFrame({
            'brand': [brand],
            'fueltype': [fueltype],
            'aspiration': [aspiration],
            'carbody': [carbody],
            'drivewheel': [drivewheel],
            'enginelocation': [enginelocation],
            'enginetype': [enginetype],
            'fuelsystem': [fuelsystem],
            'wheelbase': [wheelbase],
            'carlength': [carlength],
            'carwidth': [carwidth],
            'curbweight': [curbweight],
            'cylindernumber': [cylindernumber],
            'enginesize': [enginesize],
            'boreratio': [boreratio],
            'horsepower': [horsepower],
            'citympg': [citympg],
            'highwaympg': [highwaympg]
        })

        prediction = pipeline.predict(input_data)
        st.success(f":money_with_wings: Predicted Price: **${prediction[0]:.2f}**")
    except Exception as e:
        st.error(f":x: Prediction failed: {e}")