import os
import gradio as gr
import joblib
import pandas as pd

# Load trained model
model = joblib.load("my_loan_approval_model.pkl")


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

    data = pd.DataFrame({
        " no_of_dependents": [no_of_dependents],
        " education": [education],
        " self_employed": [self_employed],
        " income_annum": [income_annum],
        " loan_amount": [loan_amount],
        " loan_term": [loan_term],
        " cibil_score": [cibil_score],
        " residential_assets_value": [residential_assets_value],
        " commercial_assets_value": [commercial_assets_value],
        " luxury_assets_value": [luxury_assets_value],
        " bank_asset_value": [bank_asset_value],
    })

    prediction = model.predict(data)[0]

    return "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"


demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of Dependents"),

        gr.Dropdown(
            choices=["Graduate", "Not Graduate"],
            label="Education"
        ),

        gr.Dropdown(
            choices=["Yes", "No"],
            label="Self Employed"
        ),

        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term (Years)"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value"),
    ],

    outputs=gr.Textbox(label="Prediction"),

    title="Loan Approval Prediction",

    description="Enter applicant details to predict loan approval."
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
