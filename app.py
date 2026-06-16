import pickle
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

# Page Config

st.set_page_config(
    page_title="IMDb Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Constants

VOCAB_SIZE = 10000
MAX_LEN = 200

# Custom Standardization

def custom_standardization(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "<br />", " ")
    text = tf.strings.regex_replace(text, "[^a-zA-Z ]", "")
    return text

# Load Vocabulary


@st.cache_resource
def load_vectorizer():

    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    vectorizer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LEN
    )

    vectorizer.set_vocabulary(vocab)

    return vectorizer

# Load Model

@st.cache_resource
def load_rnn_model():
    return load_model("imdb_model.keras")

vectorizer = load_vectorizer()
model = load_rnn_model()

# Prediction

def predict_sentiment(review):

    review_tensor = tf.constant([review])

    review_vector = vectorizer(review_tensor)

    prediction = model.predict(review_vector, verbose=0)[0][0]

    sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"

    return sentiment, prediction

# UI

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.markdown(
"""
This application predicts whether a movie review is **Positive** or **Negative**
using a **Simple RNN** built with TensorFlow/Keras.
"""
)

review = st.text_area(
    "Enter your movie review",
    height=200,
    placeholder="Example: This movie was absolutely fantastic..."
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        sentiment, score = predict_sentiment(review)

        st.subheader("Prediction")

        st.metric("Prediction Score", f"{score:.4f}")

        if sentiment.startswith("Positive"):
            st.success(sentiment)
        else:
            st.error(sentiment)

st.markdown("---")
st.caption("Developed using TensorFlow • Keras • Streamlit")