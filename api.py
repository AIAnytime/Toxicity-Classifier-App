from fastapi import FastAPI
import pickle 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()

# load the Tfidf and model
tfidf = pickle.load(open("tf_idf.pkt", "rb"))
nb_model = pickle.load(open("toxicity_model.pkt", "rb"))

#endpoint
@app.post("/predict")
async def predict(text: str):
    #transform the inptut to Tfidf vectors 
    text_tfidf = tfidf.transform([text]).toarray()
    
    #predict the class of the input text
    prediction = nb_model.predict(text_tfidf)
    
    #map the predicted class to a string
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    
    #Return the prediction in a JSON response
    return {
        "text":text,
        "class":class_name
    }
    

