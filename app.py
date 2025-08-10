import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import os
import base64
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not present
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(os.path.join(nltk_data_dir, "corpora/stopwords")):
    nltk.download('stopwords', download_dir=nltk_data_dir)

@st.cache_resource
def load_stopwords():
    return set(stopwords.words('english'))

@st.cache_resource
def load_model_and_vectorizer():
    with open('logistic_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

@st.cache_data
def load_data():
    return pd.read_csv("Twitter_Data.csv")

def get_base64_bg(file_path):
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

def preprocess(text, stop_words):
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'http\S+|@\w+|#|<.*?>|\d+|[^\w\s]', '', text)
    text = text.lower().strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predict_sentiment(text, model, vectorizer, stop_words):
    cleaned = preprocess(text, stop_words)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)[0]
    return {1: "Positive", 0: "Neutral", -1: "Negative"}.get(result, "Unknown")

def show_login():
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Please enter both username and password")

def show_signup():
    st.subheader("📝 Sign Up")
    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")
    if st.button("Sign Up"):
        if username and password:
            st.session_state.logged_in = True
            st.success("Signup successful!")
        else:
            st.error("Please fill in all fields")

def show_sentiment_analyzer():
    st.title("📊 Public Sentiment Analyzer")
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    try:
        df = load_data()
        if 'clean_text' not in df.columns:
            st.error("Missing 'clean_text' column in dataset")
            return
        tweet_list = df['clean_text'].dropna().unique().tolist()[:1000]

        with st.form("form"):
            selected_text = st.selectbox("Choose a sample tweet", tweet_list)
            user_input = st.text_area("Or write your own...", height=150)
            if st.form_submit_button("Analyze"):
                input_text = user_input.strip() or selected_text
                sentiment = predict_sentiment(input_text, model, vectorizer, stop_words)
                color = {"Positive": "#4CAF50", "Neutral": "#999999", "Negative": "#f44336"}[sentiment]
                st.markdown(
                    f"""
                    <div style="background-color:{color}; padding:20px; border-radius:10px; margin-top:20px">
                        <h4 style="color:white;">Sentiment: {sentiment}</h4>
                        <p style="color:white;">{input_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.set_page_config(page_title="PolicyPulse", layout="centered")

    bg_image = get_base64_bg("image_n.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        with tab1:
            show_login()
        with tab2:
            show_signup()
    else:
        show_sentiment_analyzer()

if __name__ == "__main__":
    main()
