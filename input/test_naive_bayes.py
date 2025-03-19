import json
import numpy as np
import tldextract
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Data
def load_data(legit_file, phishing_file):
    with open(legit_file, "r") as f:
        legit_urls = json.load(f)
    with open(phishing_file, "r") as f:
        phishing_urls = json.load(f)

    # Labels: Legitimate (0), Phishing (1)
    urls = legit_urls + phishing_urls
    labels = [0] * len(legit_urls) + [1] * len(phishing_urls)

    return urls, labels

# Feature Extraction: Convert URL into numerical features
def extract_features(urls):
    features = []
    
    for url in urls:
        ext = tldextract.extract(url)
        domain = ext.domain
        suffix = ext.suffix
        subdomain = ext.subdomain

        features.append([
            len(url),                          # URL length
            len(domain),                       # Domain length
            len(subdomain),                    # Subdomain length
            len(suffix),                       # TLD length
            url.count('.'),                    # Count of dots in URL
            url.count('-'),                    # Count of hyphens
            url.count('/'),                    # Count of slashes
            url.count('@'),                    # Presence of '@' (phishing indicator)
            sum(c.isdigit() for c in url),     # Count of digits
            "login" in url.lower(),            # Contains 'login'
            "bank" in url.lower(),             # Contains 'bank'
            "secure" in url.lower(),           # Contains 'secure'
            "verify" in url.lower(),           # Contains 'verify'
        ])

    return np.array(features)

# Load dataset
legit_file = "data_legitimate_36400.json"
phishing_file = "data_phishing_37175.json"
urls, labels = load_data(legit_file, phishing_file)

# Convert URLs into feature vectors
X = extract_features(urls)
y = np.array(labels)

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Function to predict new URLs
def predict_url(url):
    features = extract_features([url])
    prediction = model.predict(features)[0]
    return "Phishing" if prediction == 1 else "Legitimate"

# Test a new URL
test_url = "http://secure-login-bank.com"
print(f"Prediction for {test_url}: {predict_url(test_url)}")
