from fastapi import FastAPI
import joblib
import pandas as pd
import os

# Load the pipeline once at the start
pipeline_path = os.path.join(os.getcwd(), "app", "credit_score_pipeline.pkl")
pipeline = joblib.load(pipeline_path)

def predict_credit_score(Age, Income, Education, Gender, Marital_Status, Home_Ownership, Number_of_children):
    # Create the input DataFrame
    X_new = pd.DataFrame([{
        "Income": Income,
        "Number of Children": Number_of_children,
        "Age": Age,
        "Education": Education,
        "Gender": Gender,
        "Home Ownership": Home_Ownership,
        "Marital Status": Marital_Status
    }])


    pred = pipeline.predict(X_new)
    reverse_map = {0: "Low", 1: "Average", 2: "High"}
    predicted_label = reverse_map[pred[0]]
    return predicted_label


app = FastAPI()

@app.get("/")
def index():
    return {"health": "ok"}

@app.post("/predict")
def predict(Age: int, Income: float, Education: str, Gender: str, Marital_Status: str, Home_Ownership: str, Number_of_children: int):
    prediction = predict_credit_score(Age, Income, Education, Gender, Marital_Status, Home_Ownership, Number_of_children)
    return {"prediction": f"The predicted credit score is {prediction}"}