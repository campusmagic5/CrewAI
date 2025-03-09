import pandas as pd
import joblib

# Load the trained model
model_filename = "best_model.pkl"

try:
    model = joblib.load(model_filename)
    print(f"✅ Model '{model_filename}' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Define new data for prediction (ensure it matches the training format)
new_data = pd.DataFrame([
    {"feature1": 45, "feature2": 20, "feature3": "B"},
    {"feature1": 70, "feature2": 15, "feature3": "A"},
])

# Preprocess the new data (ensure same encoding as training)
new_data = pd.get_dummies(new_data)  # Convert categorical variables

# Align new data with model's expected feature columns
expected_columns = model.feature_names_in_
for col in expected_columns:
    if col not in new_data.columns:
        new_data[col] = 0  # Add missing columns with default 0

# Reorder columns to match model input
new_data = new_data[expected_columns]

# Make predictions
predictions = model.predict(new_data)

# Print predictions
print("\n🔮 Predictions:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Predicted class ➝ {pred}")
