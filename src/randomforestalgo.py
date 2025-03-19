
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(legitimate_file, phishing_file):
    # Load legitimate data
    with open(legitimate_file, 'r') as f:
        legitimate_urls = json.load(f)
    
    # Load phishing data
    with open(phishing_file, 'r') as f:
        phishing_urls = json.load(f)
    
    # Create a list of URLs with labels
    data = []
    for url in legitimate_urls:
        data.append({
            'url': url,
            'label': 'legitimate'
        })
    for url in phishing_urls:
        data.append({
            'url': url,
            'label': 'phishing'
        })
    
    return data

def extract_features(url_data):
    # Simple feature extraction (can be extended with more sophisticated features)
    features = []
    for url in url_data:
        # Example features:
        # 1. Length of the URL
        # 2. Number of hyphens in the URL
        # 3. Number of dots in the URL
        # 4. Contains 'login' or 'secure' (common in phishing URLs)
        features.append([
            len(url['url']),
            url['url'].count('-'),
            url['url'].count('.'),
            1 if 'login' in url['url'].lower() or 'secure' in url['url'].lower() else 0
        ])
    return np.array(features)

def main():
    # Specify the paths to your JSON files
    legitimate_file = '../input/data_legitimate_36400.json'
    phishing_file = '../input/data_phishing_37175.json'
    
    # Load and prepare the data
    data = load_and_prepare_data(legitimate_file, phishing_file)
    
    # Extract features
    X = extract_features(data)
    
    # Prepare the target variable
    le = LabelEncoder()
    y = le.fit_transform([item['label'] for item in data])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Use the model to make predictions on new data
    new_urls = ["http://example.com", "apple-iforget.com"]
    new_features = extract_features([{'url': url} for url in new_urls])
    predictions = model.predict(new_features)
    
    for url, pred in zip(new_urls, predictions):
        print(f"URL: {url}, Prediction: {le.inverse_transform([pred])[0]}")

if __name__ == "__main__":
    main()