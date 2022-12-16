from autocorrect import Speller
import string
import spacy
from nltk.corpus import stopwords
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import random
import json
from pickle import load

with open('intents.json') as file:
    data = json.load(file)

tags = []
inputs = []
responses = {}

for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['patterns']:
        inputs.append(line)
        tags.append(intent['tag'])

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree= "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

#Tokenization
def tokenizer(words):
    output = [token for token in nlp(words)]
    return output

#Stop word removal:
sw = stopwords.words('english')
for x in ['what','who','how','do']:
    sw.remove(x)
def remove_stopwords(text):
    output= [word for word in text if word.text not in sw]
    return output

#Lemmatization
def lemmatizer(token):
    output= [lem.lemma_ for lem in token]
    return output

tokeniser = load(open('tokeniser.pkl','rb'))

def ASK(query):
    try:
        #spell correction
        spell = Speller(lang='en')
        query = spell(query)

        query = query.lower()
        query = remove_punctuation(query)
        query = tokenizer(query)
        query = remove_stopwords(query)
        query = lemmatizer(query)

        query = tokeniser.texts_to_sequences(query)

        query = np.array(query).reshape(-1)
        query = pad_sequences([query], padding='post', truncating='post', maxlen=8)

        model = load_model('chatbot.h5')
        output = model.predict(query)
        output = output.argmax()

        le = load(open('le.pkl','rb'))

        response_tag = le.inverse_transform([output])[0]
        output = random.choice(responses[response_tag])

        return output

    except:
        output = "Could you please rephrase your question?"

        return output

def Suggestions():
    return random.choice(inputs)
