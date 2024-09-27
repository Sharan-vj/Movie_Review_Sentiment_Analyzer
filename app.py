import numpy as np
import streamlit as st
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


# Load the trained model
model_path = Path(__file__).parent / 'model' / 'IMDB_model.h5'
model = tf.keras.models.load_model(model_path)


max_words = 10000
max_len = 500
tokenizer = Tokenizer(num_words=max_words)


(x_train, _), (_, _) = tf.keras.datasets.imdb.load_data(num_words=max_words)
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Function to decode the review (convert integer indices to words)
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# Function to preprocess the review for the model
def preprocess_review(review):
    review_seq = tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_seq, maxlen=max_len)
    return review_pad

# Streamlit App UI
st.title("Movie Review Sentiment Analyzer")
st.write("Enter a movie review, and this app will predict if it's **positive** or **negative**.")

# Input text box for the review
review_input = st.text_area("Your Review:", height=150)

# Button for prediction
if st.button('Analyze Sentiment'):
    if review_input:
        # Preprocess the review
        preprocessed_review = preprocess_review(review_input)
        
        # Make prediction using the model
        prediction = model.predict(preprocessed_review)
        
        # Determine the sentiment
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        # Display the result
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Prediction confidence: {prediction[0][0]:.2f}")
    else:
        st.write("Please enter a review to analyze.")
