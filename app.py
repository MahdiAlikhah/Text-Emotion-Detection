import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

model = joblib.load("emotion.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    vect_text = vectorizer.transform([docx])  
    result = model.predict(vect_text)
    return result[0] 

def main():
    st.title("Text-Emotion")
    with st.form("my_form"):
        raw_text = st.text_input("Enter your text here:")
        process = st.form_submit_button(label="Process")
    
    if process:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)
            
        with col2:
            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.write(f"{prediction}: {emoji_icon}")



if __name__ == '__main__':
    main()              