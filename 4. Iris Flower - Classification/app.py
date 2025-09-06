import joblib
import streamlit as st
import pandas as pd

# Load the pipeline
pipeline = joblib.load('iris_flower_classification.pkl')

# Streamlit app
st.title(":cherry_blossom: Iris Flower Classification")
st.write("Predict the species of an iris flower based on its features.")
st.image("https://content.codecademy.com/programs/machine-learning/k-means/iris.svg", width='stretch')

# Create input form
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

with col2:
    petal_length = st.slider("Petal Length (cm)", min_value=0.5, max_value=7.5, value=4.3, step=0.1)
    petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=3.0, value=1.3, step=0.1)

# Predict button
if st.button("Predict Species", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })
    prediction = pipeline.predict(input_data)
    species = prediction[0]
    
    # Display prediction success
    st.success(f"🎯 Predicted Species: **{species.title()}**")
    
    # Species information
    species_info = {
        'setosa': {
            'image': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/1365px-Irissetosa1.jpg',
            'description': 'Iris setosa is a species of flowering plant in the iris family. It is native to Alaska, Canada, and the northeastern United States.',
            'characteristics': '- Distinctive narrow petals\n- Bright purple-blue color\n- Most easily distinguishable species\n- Grows in wet, marshy areas',
            'range': 'Alaska, Canada, northeastern United States'
        },
        'versicolor': {
            'image': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Blue_Flag%2C_Ottawa.jpg/1024px-Blue_Flag%2C_Ottawa.jpg',
            'description': 'Iris versicolor, commonly known as the blue flag iris, is a species of flowering plant native to North America.',
            'characteristics': '- Medium-sized petals\n- Blue to purple color with white markings\n- Intermediate characteristics between setosa and virginica\n- Grows in wetlands and along water edges',
            'range': 'Eastern and central North America'
        },
        'virginica': {
            'image': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1024px-Iris_virginica_2.jpg',
            'description': 'Iris virginica, commonly known as the Virginia iris, is a species of flowering plant native to the eastern United States.',
            'characteristics': '- Largest petals among the three species\n- Deep purple to blue color\n- Most complex flower structure\n- Grows in moist woodlands and meadows',
            'range': 'Eastern United States'
        }
    }
    
    # Collapsible species information
    with st.expander(f"🌺 Learn more about {species.title()}", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(species_info[species]['image'], caption=f"Iris {species.title()}", width='stretch')
        
        with col2:
            st.markdown(f"**Common Name:** Iris {species}")
            st.markdown(f"**Native Range:** {species_info[species]['range']}")
            st.markdown("**Description:**")
            st.write(species_info[species]['description'])
            st.markdown("**Key Characteristics:**")
            st.write(species_info[species]['characteristics'])
    
    # Display input values used for prediction
    with st.expander("📊 **Input Values Used for Prediction:**", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Sepal Measurements:**")
            st.markdown(f"- Length: {sepal_length} cm")
            st.markdown(f"- Width: {sepal_width} cm")
        
        with col2:
            st.markdown(f"**Petal Measurements:**")
            st.markdown(f"- Length: {petal_length} cm")
            st.markdown(f"- Width: {petal_width} cm")
