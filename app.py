from flask import Flask, render_template, request, flash, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flash_messages'

# Load model safely
def load_saved_model(filename="loan_model.pkl"):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(filepath):
        print(f"Warning: Model file {filepath} not found.")
        return None
    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model_data = load_saved_model()

# Dictionary of Major Indian Banks and their approx Base Interest Rates per Loan Type
INDIAN_BANKS = {
    "SBI (State Bank of India)": {"Personal Loan": 11.5, "Home Loan": 8.5, "Education Loan": 9.5, "Business Loan": 10.5, "Vehicle Loan": 8.7},
    "HDFC Bank": {"Personal Loan": 10.5, "Home Loan": 8.7, "Education Loan": 11.0, "Business Loan": 11.5, "Vehicle Loan": 9.0},
    "ICICI Bank": {"Personal Loan": 10.8, "Home Loan": 8.75, "Education Loan": 10.5, "Business Loan": 11.0, "Vehicle Loan": 8.9},
    "Axis Bank": {"Personal Loan": 10.49, "Home Loan": 8.75, "Education Loan": 11.25, "Business Loan": 11.2, "Vehicle Loan": 9.2},
    "Punjab National Bank": {"Personal Loan": 10.4, "Home Loan": 8.4, "Education Loan": 9.2, "Business Loan": 10.0, "Vehicle Loan": 8.5},
    "Bank of Baroda": {"Personal Loan": 10.9, "Home Loan": 8.4, "Education Loan": 9.35, "Business Loan": 10.25, "Vehicle Loan": 8.6},
    "Kotak Mahindra Bank": {"Personal Loan": 10.99, "Home Loan": 8.75, "Education Loan": 11.5, "Business Loan": 11.5, "Vehicle Loan": 9.2},
    "Canara Bank": {"Personal Loan": 10.6, "Home Loan": 8.5, "Education Loan": 9.5, "Business Loan": 10.1, "Vehicle Loan": 8.8},
    "Union Bank of India": {"Personal Loan": 10.7, "Home Loan": 8.35, "Education Loan": 9.4, "Business Loan": 9.9, "Vehicle Loan": 8.7},
    "IndusInd Bank": {"Personal Loan": 10.4, "Home Loan": 8.8, "Education Loan": 11.75, "Business Loan": 11.5, "Vehicle Loan": 9.4}
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    suggestions = None
    user_data = None
    emi = None
    if request.method == "POST":
        if model_data is None:
            flash("Model is not loaded. Please contact the administrator.", "error")
            return render_template("index.html")
            
        try:
            # Parse inputs robustly
            age = int(request.form.get('age'))
            income = int(request.form.get('income'))
            loan = int(request.form.get('loan'))
            credit = int(request.form.get('credit'))
            years = int(request.form.get('years'))
            dependents = int(request.form.get('dependents'))
            
            # New fields (not used by ML model, but used for EMI and UI)
            bank_name = request.form.get('bank_name', 'SBI (State Bank of India)')
            loan_type = request.form.get('loan_type', 'Personal Loan')
            try:
                interest_rate = float(request.form.get('interest_rate', 8.0))
            except ValueError:
                interest_rate = 8.0

            # Create input dictionary
            input_data = pd.DataFrame({
                'Age': [age],
                'Income': [income],
                'Loan_Amount': [loan],
                'Credit_Score': [credit],
                'Employment_Years': [years],
                'Dependents': [dependents]
            })

            # Retrieve model objects
            model = model_data['model']
            scaler = model_data['scaler']
            le = model_data['label_encoder']

            # Transform and predict
            scaled = scaler.transform(input_data)
            pred = model.predict(scaled)
            result = le.inverse_transform(pred)[0]

            prediction_text = "Eligible" if result.upper() == "APPROVED" else "Not Eligible"
            flash(f"Loan Status: {result.upper()} ({prediction_text})", "success" if result.upper() == "APPROVED" else "warning")

            user_data = {
                'age': age, 'income': income, 'loan': loan,
                'credit': credit, 'years': years, 'dependents': dependents,
                'bank_name': bank_name, 'loan_type': loan_type, 'interest_rate': interest_rate
            }

            if result.upper() == "APPROVED":
                # Calculate EMI using custom interest rate and 5 years (60 months)
                r = (interest_rate / 100) / 12
                n = 60
                emi = (loan * r * (1 + r)**n) / ((1 + r)**n - 1)
                emi = round(emi, 2)

            if result.upper() != "APPROVED":
                suggestions = []
                if credit < 700:
                    suggestions.append("Improve your credit score (aim for 700+).")
                if income > 0 and (loan / income) > 3:
                    suggestions.append("Consider reducing the requested loan amount or getting a co-applicant.")
                if years < 2:
                    suggestions.append("A longer employment history (2+ years) improves approval odds.")
                if income < 30000:
                    suggestions.append("Your current income level might be too low for the requested loan amount.")
                if not suggestions:
                    suggestions.append("Maintain a healthy financial profile, reduce existing debts, and try again later.")

        except ValueError as ve:
            flash("Invalid input detected. Please ensure all numeric fields are filled correctly.", "error")
        except Exception as e:
            flash(f"An unexpected error occurred during prediction.", "error")
            print(f"Prediction Error: {e}")

    return render_template("index.html", suggestions=suggestions, user_data=user_data, emi=emi, banks=INDIAN_BANKS)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.json
        input_data = pd.DataFrame({
            'Age': [int(data.get('age', 30))],
            'Income': [int(data.get('income', 50000))],
            'Loan_Amount': [int(data.get('loan', 100000))],
            'Credit_Score': [int(data.get('credit', 700))],
            'Employment_Years': [int(data.get('years', 5))],
            'Dependents': [int(data.get('dependents', 0))]
        })

        model = model_data['model']
        scaler = model_data['scaler']
        le = model_data['label_encoder']

        scaled = scaler.transform(input_data)
        pred = model.predict(scaled)
        
        # If the model supports predict_proba, use it
        prob = None
        if hasattr(model, "predict_proba"):
            # typically class 1 is Approved if encoded alphabetically (A, R), wait, depends on encoder.
            # let's just use the classes_ from label encoder
            probs = model.predict_proba(scaled)[0]
            pred_class = le.inverse_transform(pred)[0]
            # assume prob of the predicted class
            class_idx = list(le.classes_).index(pred_class)
            prob = round(probs[class_idx] * 100, 1)

        result = le.inverse_transform(pred)[0]
        
        return jsonify({
            "status": result.upper(),
            "probability": prob
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)