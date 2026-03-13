import pandas as pd
import joblib
import os
import argparse

def load_saved_model(filename="loan_model.pkl"):
    """Loads the saved model and preprocessors"""
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(filepath):
        print(f"Error: Model file {filepath} not found. Please run train_model.py first.")
        return None
    
    return joblib.load(filepath)

def predict_eligibility(model_data, new_data):
    """Predicts loan eligibility for new data."""
    model = model_data['model']
    scaler = model_data['scaler']
    le = model_data['label_encoder']
    
    # Scale numerical features
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction_encoded = model.predict(new_data_scaled)
    
    # Decode prediction
    prediction = le.inverse_transform(prediction_encoded)
    
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Bank Loan Eligibility")
    parser.add_argument('--age', type=int, default=30, help="Age of applicant")
    parser.add_argument('--income', type=int, default=60000, help="Annual income")
    parser.add_argument('--loan', type=int, default=150000, help="Requested loan amount")
    parser.add_argument('--credit', type=int, default=700, help="Credit score")
    parser.add_argument('--years', type=int, default=5, help="Years of employment")
    parser.add_argument('--dependents', type=int, default=2, help="Number of dependents")
    
    args = parser.parse_args()
    
    model_data = load_saved_model()
    
    if model_data:
        # Create a DataFrame for the new data
        new_customer = pd.DataFrame({
            'Age': [args.age],
            'Income': [args.income],
            'Loan_Amount': [args.loan],
            'Credit_Score': [args.credit],
            'Employment_Years': [args.years],
            'Dependents': [args.dependents]
        })
        
        print("\nApplicant Information:")
        print(new_customer.to_string(index=False))
        print("-" * 30)
        
        result = predict_eligibility(model_data, new_customer)
        print(f"\nLoan Prediction Result: {result.upper()}")
