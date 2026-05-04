import streamlit as st
import joblib
import re

# Page configuration
st.set_page_config(page_title="Sentiment AI by Hafiz Zuhaib", page_icon="🤖")

# 1. Load the "Intelligence" (Saved Model & Vectorizer)
@st.cache_resource # Keeps it in memory so it's fast
def load_assets():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except:
    st.error("Model files not found! Please save your model/vectorizer first.")

# 2. UI Layout
st.title("🧠 Twitter Sentiment Predictor")
st.markdown("""
Welcome to my NLP Project! This model was trained on **1.6 Million Tweets** using a 
Logistic Regression algorithm and TF-IDF Vectorization.
""")

# 3. User Input
user_text = st.text_area("Paste a sentence or paragraph here to check sentiment:", 
                         placeholder="Example: I am really enjoying this machine learning journey!")

if st.button("Predict Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Step A: Clean
        from task_4 import clean_dataset # Importing our logic
        cleaned = clean_dataset(user_text)
        
        # Step B: Vectorize
        vectorized_text = vectorizer.transform([cleaned])
        
        # Step C: Predict Sentiment and Probability
        prediction = model.predict(vectorized_text)
        probs = model.predict_proba(vectorized_text) # The "Confidence" math
        
        # UI Results
        st.divider()
        sentiment = "Positive" if prediction[0] == 4 else "Negative"
        confidence = max(probs[0]) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Sentiment", sentiment)
        with col2:
            st.metric("Confidence Score", f"{confidence:.2f}%")
            
        st.write(f"**Cleaned Tokens used for math:** `{cleaned}`")
