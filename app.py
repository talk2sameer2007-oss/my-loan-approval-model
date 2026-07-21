import os
import joblib
import pandas as pd
import gradio as gr

# Load the trained model
model = joblib.load("Loan_Prediction_Model.pkl")


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
    # Create DataFrame in the SAME ORDER as training
    input_data = pd.DataFrame([[
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
        bank_asset_value
    ]], columns=[
        " no_of_dependents",
        " education",
        " self_employed",
        " income_annum",
        " loan_amount",
        " loan_term",
        " cibil_score",
        " residential_assets_value",
        " commercial_assets_value",
        " luxury_assets_value",
        " bank_asset_value"
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "✅ Loan Approved"
    else:
        return "❌ Loan Rejected"


interface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Dropdown(
            choices=[0, 1],
            label="Education (0 = Graduate, 1 = Not Graduate)"
        ),
        gr.Dropdown(
            choices=[0, 1],
            label="Self Employed (0 = No, 1 = Yes)"
        ),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Loan Approval Prediction",
    description="Enter applicant details to predict whether the loan will be approved using the trained Random Forest model."
)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
