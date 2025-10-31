import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load dataset
data = pd.read_csv("engine_data.csv")

# --- Adjust columns based on your dataset structure ---
# Assuming 'target' is the label column (e.g., maintenance_days_left)
X = data.drop('Engine Condition', axis=1)
y = data['Engine Condition']


# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train GBM model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model.pkl successfully!")
