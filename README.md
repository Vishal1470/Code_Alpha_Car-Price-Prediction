🚗 Car Price Prediction App

A machine learning web app built with Streamlit that predicts the resale price of a car based on user inputs like brand, engine size, mileage, model year, and photo condition.
The app supports car image uploads and displays popular car brand logos for a professional look.

🧠 Features

🔍 Car Price Prediction using trained ML regression model

🖼️ Car Photo Analyzer (simulated visual condition scoring)

🧮 Automatic Feature Encoding & Alignment

💾 Joblib-based model loading

🏷️ Brand logo display for top car brands

🌐 Simple, interactive Streamlit dashboard

📁 Project Structure
Code_Alpha_Car Price Prediction/

│

├── dataset/

│   └── car_data.csv

│

├── models/

│   └── car_price_model.pkl

│

├── scripts/

│   ├── model_training.py

│   └── data_cleaning.py

│

├── app.py

├── requirements.txt

└── README.md

⚙️ Installation Steps

Clone or Download the Project

git clone https://github.com/Vishal1470/car-price-prediction.git
cd "Code_Alpha_Car Price Prediction"


Install Dependencies

pip install -r requirements.txt


Train the Model (if not already trained)

python scripts/model_training.py


Run the App

streamlit run app.py


Open in browser → http://localhost:8501

📊 Model Info

Algorithm: Linear Regression / RandomForestRegressor (based on training code)

Trained On: Cleaned dataset with features — brand, transmission, mileage, engine size, age

Saved Using: joblib (models/car_price_model.pkl)

📷 Car Photo Analyzer

Allows users to upload a car photo (.jpg, .jpeg, .png)

Simulates visual condition scoring (0–1 scale)

Adjusts predicted price accordingly

Future version can use CNN (e.g., ResNet50) for real visual assessment

🧾 Requirements

Minimal dependencies:

streamlit
pandas
numpy
scikit-learn
joblib
Pillow

🧑‍💻 Author

Vishal Baburao Patil
B.Tech CSE — G H Raisoni College of Engineering and Management, Jalgaon
