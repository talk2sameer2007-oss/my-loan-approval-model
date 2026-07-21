import os
import joblib
import pandas as pd
import gradio as gr

# ==========================================================
# Load Model
# ==========================================================

try:
    model = joblib.load("Loan_Prediction_Model.pkl")
except Exception as e:
    print(e)
    model = None


# ==========================================================
# Prediction Function
# ==========================================================

def predict_loan(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):

    if model is None:
        return "❌ Model not loaded."

    try:

        education = 1 if education == "Graduate" else 0
        self_employed = 1 if self_employed == "Yes" else 0

        input_df = pd.DataFrame([{
            " no_of_dependents": int(no_of_dependents),
            " education": education,
            " self_employed": self_employed,
            " income_annum": float(income_annum),
            " loan_amount": float(loan_amount),
            " loan_term": int(loan_term),
            " cibil_score": int(cibil_score),
            " residential_assets_value": float(residential_assets_value),
            " commercial_assets_value": float(commercial_assets_value),
            " luxury_assets_value": float(luxury_assets_value),
            " bank_asset_value": float(bank_asset_value),
        }])

        prediction = model.predict(input_df)[0]

        if prediction == 0 or prediction == " Approved":
            return """
✅ Loan Approved

Congratulations!
The model predicts that the loan is likely to be approved.
"""

        return """
❌ Loan Rejected

The model predicts that the loan is likely to be rejected.
"""

    except Exception as e:
        return f"Prediction Error:\n{e}"


# ==========================================================
# Description
# ==========================================================

DESCRIPTION = """
# 🏦 Loan Approval Prediction

# 👩‍💻 Developer Details

*Name:* Sameer

*College:*  
Panipat Institute of Engineering and Technology

---

# 📌 Project

Loan Approval Prediction using Random Forest Classifier

---

# 🛠️ Technology Used

- Python
- Pandas
- Scikit-Learn
- Random Forest
- Joblib
- Gradio

---

### Input Features

- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value
"""

# ==========================================================
# Gradio Interface
# ==========================================================

demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Dropdown(["Graduate", "Not Graduate"], label="Education"),
        gr.Dropdown(["Yes", "No"], label="Self Employed"),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value"),
    ],
    outputs=gr.Textbox(label="Prediction", lines=5),
    title="🏦 Loan Approval Prediction",
    description=DESCRIPTION,
)

# ==========================================================
# Launch
# ==========================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
