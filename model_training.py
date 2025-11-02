import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# ==========================================================
# Load Dataset
# ==========================================================
file_path = "dataset/car_data.csv"
df = pd.read_csv(file_path)

# Create new feature 'age' from model_year
df['age'] = 2025 - df['model_year']

# Features and Target
X = df[['brand', 'age', 'horsepower', 'mileage', 'engine_size', 'transmission']]
y = df['price']

# ==========================================================
# Preprocessing and Model Pipeline
# ==========================================================
categorical_features = ['brand', 'transmission']
numeric_features = ['age', 'horsepower', 'mileage', 'engine_size']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model.fit(X_train, y_train)

# ==========================================================
# Save Model
# ==========================================================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/car_price_model.pkl")
print("✅ Model trained and saved successfully at models/car_price_model.pkl")
