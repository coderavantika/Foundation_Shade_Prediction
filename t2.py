# Step 1: Data Preparation and Model Training
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
file_path = 'undertones.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Clean and preprocess data
data = data.dropna(how='all').reset_index(drop=True)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data.drop(columns=['Skin Tone'], errors='ignore')

# Standardize text formatting
data['Undertone'] = data['Undertone'].str.strip().str.capitalize()
data['Shade Depth'] = data['Shade Depth'].str.strip().str.capitalize()

# Merge similar categories (if needed)
data['Undertone'] = data['Undertone'].replace({
    'Peachy': 'Peach',
})

# Encode categorical columns
encoder_undertone = LabelEncoder()
encoder_shade_depth = LabelEncoder()

data['Undertone'] = encoder_undertone.fit_transform(data['Undertone'])
data['Shade Depth'] = encoder_shade_depth.fit_transform(data['Shade Depth'])

# Features and target variable
X = data[['Undertone', 'Shade Depth']]
y = data['Shade Name']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save the model and encoders
joblib.dump(model, 'foundation_shade_model.pkl')
joblib.dump(encoder_undertone, 'encoder_undertone.pkl')
joblib.dump(encoder_shade_depth, 'encoder_shade_depth.pkl')
print("Model and encoders saved successfully!")

# Step 2: Streamlit Application for Prediction
import streamlit as st

st.title("Foundation Shade Predictor")

# Load the Model and Encoders
try:
    # Load the trained model
    model = joblib.load('foundation_shade_model.pkl')
    st.write("Checkpoint 1: Model loaded successfully.")

    # Load encoders
    encoder_undertone = joblib.load('encoder_undertone.pkl')
    encoder_shade_depth = joblib.load('encoder_shade_depth.pkl')
    st.write("Checkpoint 2: Encoders loaded successfully.")

except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()  # Stop execution if loading fails

# Display encoder options to user
valid_undertones = list(encoder_undertone.classes_)
valid_shade_depths = list(encoder_shade_depth.classes_)

# User Input
undertone = st.selectbox("Select your undertone:", options=valid_undertones)
shade_depth = st.selectbox("Select your shade depth:", options=valid_shade_depths)

# Predict Button
if st.button("Predict Foundation Shade"):
    try:
        # Validate and Encode Inputs
        if undertone not in valid_undertones:
            st.error("Invalid undertone selected. Please choose from the available options.")
        elif shade_depth not in valid_shade_depths:
            st.error("Invalid shade depth selected. Please choose from the available options.")
        else:
            encoded_undertone = encoder_undertone.transform([undertone])[0]
            encoded_shade_depth = encoder_shade_depth.transform([shade_depth])[0]

            # Prepare data for prediction
            new_data = pd.DataFrame({
                'Undertone': [encoded_undertone],
                'Shade Depth': [encoded_shade_depth]
            })

            # Make Prediction
            predicted_shade = model.predict(new_data)
            st.success(f"Your predicted foundation shade is: **{predicted_shade[0]}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.write("Ensure the uploaded encoders and models are correctly aligned with the dataset.")
