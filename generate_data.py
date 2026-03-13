import pandas as pd
import numpy as np
import random

def generate_loan_data(num_samples=1000):
    """Generates a synthetic dataset for bank loan eligibility."""
    np.random.seed(42)
    random.seed(42)
    
    data = []
    for _ in range(num_samples):
        # Generate feature values
        age = np.random.randint(21, 65)
        income = np.random.randint(20000, 150000)
        loan_amount = np.random.randint(5000, 500000)
        credit_score = np.random.randint(300, 850)
        employment_years = np.random.randint(0, 40)
        dependents = np.random.randint(0, 5)
        
        # Determine loan status based on simple logical rules
        # High income and good credit score -> Approved
        # Low income, high loan, or low credit score -> Rejected
        
        score = 0
        if income > 50000:
            score += 2
        if credit_score > 650:
            score += 3
        if employment_years > 2:
            score += 1
        if loan_amount < income * 3:
            score += 2
            
        # Add some random noise
        score += np.random.randint(-1, 2)
        
        loan_status = 'Approved' if score >= 5 else 'Rejected'
        
        data.append([age, income, loan_amount, credit_score, employment_years, dependents, loan_status])
        
    df = pd.DataFrame(data, columns=['Age', 'Income', 'Loan_Amount', 'Credit_Score', 'Employment_Years', 'Dependents', 'Loan_Status'])
    return df

if __name__ == "__main__":
    print("Generating synthetic loan data...")
    df = generate_loan_data(1000)
    df.to_csv("bank_loan_data.csv", index=False)
    print(f"Data generated and saved to bank_loan_data.csv. Shape: {df.shape}")
    print("\nSample Data:")
    print(df.head())
