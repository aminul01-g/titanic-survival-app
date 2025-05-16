import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("titanic_model.pkl")

# Prediction function
def predict_survival(pclass, sex, age, fare):
    sex_val = 1 if sex == "male" else 0
    data = np.array([[pclass, sex_val, age, fare]])
    prediction = model.predict(data)[0]
    return "✅ Survived" if prediction == 1 else "❌ Did not survive"

# Gradio interface
app = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Number(label="Ticket Class (1-3)"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="Age"),
        gr.Number(label="Fare Paid")
    ],
    outputs="text",
    title="Titanic Survival Predictor",
    description="Enter passenger info to see if they'd survive the Titanic!"
)

app.launch()
