import streamlit as st
import torch
import os
import gdown
from transformers import BertTokenizer, BertForSequenceClassification


DRIVE_FOLDER_ID = "https://drive.google.com/drive/folders/1r0car95c5vv84Z5_qLobsKRvKCWqtkJ7?usp=sharing"
MODEL_DIR = "model"

if not os.path.exists(MODEL_DIR):
    st.write("Downloading fine-tuned BERT model from Google Drive...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    gdown.download_folder(id=DRIVE_FOLDER_ID, output=MODEL_DIR, quiet=False, use_cookies=False)
    st.success("✅ Model downloaded successfully!")


@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load_model()
emotion_labels = ['anger', 'joy', 'neutral', 'sadness']


st.title("Emotion Detection from Text (Fine-tuned BERT)")
st.markdown("Predict emotions like joy, sadness, anger, and neutral from any text input.")

text = st.text_area("Enter your sentence:", height=120)

if st.button("Predict Emotion"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
        st.success(f"Predicted Emotion: {emotion_labels[pred]}")
    else:
        st.warning("Please enter a sentence to analyze.")
