import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^A-Za-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.lower().split()

    filtered_words = [word for word in words if word not in stop_words]

    return ' '.join(filtered_words)