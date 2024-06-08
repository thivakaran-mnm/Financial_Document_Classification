import pickle
import re
import nltk
import pickle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import streamlit as st

# download the necessary NLTK data files(once)
#nltk.download('stopwords')
#nltk.download('punkt')

def extract_text_from_file(html_file):
    
    # Extracting text from html file
    soup = BeautifulSoup(html_file, 'html.parser')
    ext_text = soup.get_text(separator=' ')
    return ext_text

def clean_text(text):

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back into a single string
    clean_text = ' '.join(tokens)

    return clean_text

def embed_text(text):

    # Load the Vectorizer model
    with open("C:\\Users\\hp\\tfidf_vectorizer_1.pkl","rb") as f2:
        tfidf_vectorizer = pickle.load(f2)

    # Transforming cleaned text to vector
    text_new = tfidf_vectorizer.transform([text])
    return text_new

def predict_category(text_new):

    # Load the model
    with open("C:\\Users\\hp\\Findoc_Classification_model.pkl","rb") as f1:
        class_model= pickle.load(f1)
    
    predictions = class_model.predict(text_new)

    if predictions == 0:
        pred = "Balance Sheets"
    elif predictions == 1:
        pred = "Cash Flow"
    elif predictions == 2:
        pred = "Income Statement"
    elif predictions == 3:
        pred = "Notes"
    elif predictions == 4:
        pred = "Others"

    return pred

# streamlit part
st.title("Financial Statement Predictor")

try:
    # File uploader
    uploaded_file = st.file_uploader("Upload an HTML file", type="html")
    # Extract text from html file
    ext_text = extract_text_from_file(uploaded_file)
    # Clean the extracted text
    text_new = clean_text(ext_text)
    # Embedding cleaned text to vector
    text=embed_text(text_new)
    # Predict by passing vector
    category = predict_category(text)
    # Display the Predicted Category
    st.write("## :green[**Predicted Category :**]", category)

except TypeError as e:
    print("Upload the Financial Statement")
    pass

