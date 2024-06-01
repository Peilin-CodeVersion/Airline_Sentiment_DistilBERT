import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Ensure NLTK data is downloaded
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Load model and tokenizer
model_dir = 'Peilin-CodeVersion/Airline_Sentiment_Classifier_DistilBERT'
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_dir)

# Set up Streamlit interface
st.title("Welcome to the Airline Sentiment Review Classifier!")
st.markdown("""
    The Airline Sentiment Review Classifier analyzes and classifies customer review sentiments.

    It determines if the sentiment is positive, negative, or neutral and it processes text data from reviews to provide insights.

    This application was developed for airlines to understand customer feedback and improve services.
""")

# User input
user_input = st.text_area("It’s simple to kick start! Just input your airline review & our model will analyze and check the text you provided.")

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove HTML tags
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove special symbols and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    
    # Remove punctuations
    punctuations = '#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')
    
    # Convert to lowercase and split into words
    words = text.lower().split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join the words back into a sentence
    processed_text = ' '.join(filtered_words)
    
    return processed_text

def get_aspect(text):
    tagged = pos_tag(word_tokenize(text))
    aspects = []
    for i 

