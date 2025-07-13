import streamlit as st
import pandas as pd
import pickle
import string
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Prediction function
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

# Set page config
st.set_page_config(page_title="Sentiment Analyzer", layout="centered", page_icon="💬")

# Sidebar
with st.sidebar:
    st.title("💡 About the App")
    st.markdown(
        """
        This is a simple and elegant **Sentiment Analysis** app built with:
        - 🧠 Naive Bayes Classifier  
        - 🧹 NLP (stopwords, stemming, TF-IDF)  
        - 🧪 Amazon Fine Food Reviews dataset  
        """
    )
    st.markdown("---")
    st.info("Created by Vishwajeet", icon="🧑‍💻")

# Main Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📝 Product Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("### Tell us how you feel about a product...")

# Tabs
tab1, tab2 = st.tabs(["🔍 Analyze One Review", "📁 Analyze Batch (CSV)"])

# -------------------------
# SINGLE REVIEW TAB
# -------------------------
with tab1:
    st.markdown("Type a review below and click **Analyze** to get the sentiment prediction.")

    input_text = st.text_area("Your Review:", placeholder="e.g. This product is amazing, totally worth it!")

    if st.button("🔎 Analyze", use_container_width=True):
        if input_text.strip() == "":
            st.warning("⚠️ Please enter a review first.")
        else:
            sentiment = predict_sentiment(input_text)
            if sentiment == "positive":
                st.success("Sentiment: Positive 😊", icon="✅")
            else:
                st.error("Sentiment: Negative 😞", icon="❌")

# -------------------------
# BATCH CSV TAB
# -------------------------
with tab2:
    st.markdown("Upload a CSV file with a `Text` column. It will return a CSV with predicted sentiment.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Text' not in df.columns:
                st.error("CSV must contain a column named 'Text'")
            else:
                df['Cleaned'] = df['Text'].astype(str).apply(clean_text)
                X = vectorizer.transform(df['Cleaned'])
                df['Predicted_Sentiment'] = model.predict(X)

                st.success("✅ Successfully analyzed all reviews!")
                st.dataframe(df[['Text', 'Predicted_Sentiment']].head(10), use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", data=csv, file_name="sentiment_output.csv", mime='text/csv')
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
