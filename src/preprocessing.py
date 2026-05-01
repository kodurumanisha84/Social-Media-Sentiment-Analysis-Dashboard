import re
from nltk.corpus import stopwords

def clean_text(text):
    text = re.sub(r"http\S+|[^A-Za-z\s]", "", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text