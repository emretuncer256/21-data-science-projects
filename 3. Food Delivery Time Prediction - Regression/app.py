import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load the model pipeline
pipeline = joblib.load('food_delivery_time_prediction_model.pkl')

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

# Load the categorical choices
categorical_choices = extract_categorical_choices_from_pipeline(pipeline)

# Streamlit app
st.title(":motor_scooter: Food Delivery Time Prediction")
st.write("Predict the time taken to deliver food based on various features.")
st.image("https://t4.ftcdn.net/jpg/03/94/73/73/360_F_394737308_A5IJf7vijvkGWCsiCcNI1kAGWoa5g54h.jpg", use_container_width=True)

# Create input form
st.header("📝 Delivery Information")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Delivery Person Age
    delivery_person_age = st.number_input(
        "Delivery Person Age",
        min_value=15,
        max_value=70,
        value=30,
        help="Age of the delivery person (15-70 years)"
    )
    
    # Delivery Person Ratings
    delivery_person_ratings = st.slider(
        "Delivery Person Ratings",
        min_value=1.0,
        max_value=6.0,
        value=4.6,
        step=0.1,
        help="Rating of the delivery person (1.0-6.0)"
    )

with col2:
    # Type of Order
    type_of_order = st.selectbox(
        "Type of Order",
        options=categorical_choices.get('Type_of_order', ['Snack', 'Meal', 'Drinks', 'Buffet']),
        help="Type of food order"
    )
    
    # Type of Vehicle
    type_of_vehicle = st.selectbox(
        "Type of Vehicle",
        options=categorical_choices.get('Type_of_vehicle', ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']),
        help="Vehicle used for delivery"
    )

# Distance (full width)
st.subheader("📍 Distance Information")

# Distance input method selection
distance_method = st.radio(
    "Choose distance input method:",
    ["Enter distance directly", "Calculate from coordinates"],
    horizontal=True
)

if distance_method == "Enter distance directly":
    # Direct distance input
    distance = st.number_input(
        "Delivery Distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1,
        format="%.1f",
        help="Enter the distance between restaurant and delivery location in kilometers"
    )
    st.info(f"📏 Distance: {distance:.1f} km")
    
else:
    # Coordinate-based distance calculation
    col3, col4 = st.columns(2)

    with col3:
        st.write("**Restaurant Location**")
        restaurant_lat = st.number_input(
            "Restaurant Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=17.017729,
            format="%.6f",
            help="Latitude of the restaurant"
        )
        restaurant_lon = st.number_input(
            "Restaurant Longitude", 
            min_value=-180.0,
            max_value=180.0,
            value=70.231332,
            format="%.6f",
            help="Longitude of the restaurant"
        )

    with col4:
        st.write("**Delivery Location**")
        delivery_lat = st.number_input(
            "Delivery Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=17.465186,
            format="%.6f",
            help="Latitude of the delivery location"
        )
        delivery_lon = st.number_input(
            "Delivery Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=70.845702,
            format="%.6f",
            help="Longitude of the delivery location"
        )


    # Calculate distance using Haversine formula
    def calculate_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        d_lat = np.radians(lat2 - lat1)
        d_lon = np.radians(lon2 - lon1)
        a = np.sin(d_lat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    # Calculate and display distance
    distance = calculate_distance(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon)
    st.metric("Calculated Distance", f"{distance:.2f} km")

# Prediction button
if st.button("🚀 Predict Delivery Time", type="primary", use_container_width=True):
    # Prepare input data
    input_data = pd.DataFrame({
        'Delivery_person_Age': [delivery_person_age],
        'Delivery_person_Ratings': [delivery_person_ratings],
        'Type_of_order': [type_of_order],
        'Type_of_vehicle': [type_of_vehicle],
        'Distance': [distance]
    })
    
    # Make prediction
    prediction = pipeline.predict(input_data)[0]
    
    # Display results
    st.success(f"⏱️ **Predicted Delivery Time: {prediction:.1f} minutes**")
    
    # Additional insights
    st.subheader("📊 Prediction Insights")
    
    if prediction < 20:
        st.info("⚡ Fast delivery expected! This is likely a short-distance order.")
    elif prediction < 30:
        st.info("✅ Normal delivery time. Good balance of distance and efficiency.")
    elif prediction < 40:
        st.warning("⏳ Longer delivery time. Consider factors like distance or vehicle type.")
    else:
        st.warning("🐌 Extended delivery time. This might be a long-distance or challenging delivery.")
    
    # Show input summary
    with st.expander("📋 Input Summary"):
        st.write(f"**Delivery Person:** {delivery_person_age} years old, {delivery_person_ratings}⭐ rating")
        st.write(f"**Order:** {type_of_order} via {type_of_vehicle}")
        st.write(f"**Distance:** {distance:.2f} km")
        st.write(f"**Predicted Time:** {prediction:.1f} minutes")
