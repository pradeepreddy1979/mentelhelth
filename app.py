import streamlit as st
import pandas as pd
import pickle
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Page config
st.set_page_config(page_title="Mental Health Detector", layout="centered")

st.title("🧠 Mental Health Condition Detector")
st.write("Enter a sentence and the AI model will predict the mental health category.")

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
    return tokenizer, model

tokenizer, model = load_model()

# Input text
user_input = st.text_area("Enter text")

# Prediction function
def predict(text):
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    labels = {
        0: "Depression",
        1: "Suicidal",
        2: "Normal"
    }
    
    return labels[prediction]

# Button
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter text")
    
    else:
        result = predict(user_input)
        
        st.success(f"Prediction: **{result}**")