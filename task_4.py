import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure these are downloaded for the Streamlit Server
nltk.download('stopwords')
nltk.download('wordnet')

stop_word = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_dataset(text):
    # clean @mentions
    text = re.sub(r"@[a-zA-Z0-9_]+", " ", text)
    # remove url's
    text = re.sub(r"http\s+|www\s+|https\s+", " ", text, flags=re.MULTILINE)
    # remove non-letters (Note: I fixed your regex [~a-zA-Z] to [^a-zA-Z])
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    words = text.lower().split()
    clean_word = [lemmatizer.lemmatize(w) for w in words if w not in stop_word]
    return " ".join(clean_word)
