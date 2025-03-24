import json
import numpy as np
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Data
def load_data(legit_file, phishing_file):
    with open(legit_file, "r") as f:
        legit_urls = json.load(f)
    with open(phishing_file, "r") as f:
        phishing_urls = json.load(f)

    urls = legit_urls + phishing_urls
    labels = [0] * len(legit_urls) + [1] * len(phishing_urls)
    return urls, labels

# Optimized TF-IDF Feature Extraction
def extract_features_tfidf(urls):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,4), max_features=3000)
    X = vectorizer.fit_transform(urls)
    return X, vectorizer

# Load dataset
legit_file = "data_legitimate_36400.json"
phishing_file = "data_phishing_37175.json"
urls, labels = load_data(legit_file, phishing_file)

# Convert URLs into TF-IDF feature vectors
X_tfidf, vectorizer = extract_features_tfidf(urls)
y = np.array(labels)

# Train on a smaller subset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf[:30000], y[:30000], test_size=0.2, random_state=42)

# Optimized Random Forest Model
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
cv_scores = cross_val_score(model, X_tfidf, y, cv=5)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Function to predict new URLs
def predict_url(url):
    features = vectorizer.transform([url])
    prediction = model.predict(features)[0]
    return "Phishing" if prediction == 1 else "Legitimate"

# Test a new URL
test_url = "http://secure-login-bank.com"
print(f"Prediction for {test_url}: {predict_url(test_url)}")