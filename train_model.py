import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_data(filepath="bank_loan_data.csv"):
    """Loads the dataset."""
    if not os.path.exists(filepath):
        print(f"Error: Dataset {filepath} not found. Please run generate_data.py first.")
        return None
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocesses the Data."""
    print("Preprocessing data...")
    # Separate features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def train_model(X_train, y_train):
    """Trains a Random Forest Classifier."""
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, le):
    """Evaluates the model performance."""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, scaler, le, filename="loan_model.pkl"):
    """Saves the trained model and preprocessing steps"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le
    }
    joblib.dump(model_data, filename)
    print(f"Model saved completely to {filename}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        X_train, X_test, y_train, y_test, scaler, le = preprocess_data(df)
        rf_model = train_model(X_train, y_train)
        evaluate_model(rf_model, X_test, y_test, le)
        save_model(rf_model, scaler, le)
