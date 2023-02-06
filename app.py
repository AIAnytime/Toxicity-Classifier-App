import streamlit as st
import pickle 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
    return nb_model

def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

st.header("Toxicity Detection App")

st.subheader("Input your text")

text_input = st.text_input("Enter your text")

if text_input is not None:
    if st.button("Analyse"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        st.info("The result is "+ result + ".")
