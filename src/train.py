import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load data
data = pd.read_csv("data/data.csv")

# Clean data
data = data.dropna()
data = data[data["text"].str.strip() != ""]

# Features & labels
X = data["text"]
y = data["sentiment"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorizer (better version)
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Model (better version)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model trained and saved successfully!")