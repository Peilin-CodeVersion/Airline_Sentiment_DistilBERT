import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


model_dir = "DistilBERT"

# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Create a title for the app
st.title('DistilBERT Classifier')

# Create a text input for user input
user_input = st.text_input("Enter your text here")

# Create a button for getting predictions
if st.button('Predict'):
    # Tokenize the user input
    inputs = tokenizer(user_input, return_tensors='pt')

    # Get the model's prediction
    outputs = model(**inputs)

    # Get the predicted class
    predicted_class = outputs.logits.argmax(-1).item()

    # Display the predicted class
    st.write(f"The predicted class for the input is: {predicted_class}")
