import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    print("Before cleaning:", df.shape)

    # Handle missing values
    df = df.dropna()

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    print("After cleaning:", df.shape)
    return df

def split_data(df, target='price'):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)
