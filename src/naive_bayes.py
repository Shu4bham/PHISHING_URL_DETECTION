
import json
import datetime
import numpy as np

from io import StringIO
from scipy.io import arff
from traceback import format_exc
from domain_parser import domain_parser
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ns_log import NsLog
from json2arff import json2arff
from rule_extraction import rule_extraction

class machine_learning_algorithm():

    def __init__(self, algorithm, train_data_name="gsb.arff"):

        self.logger = NsLog("log")

        self.path_output_arff = "../output/arff/"
        self.path_test_output = ""

        self.json2arff_object = json2arff()
        self.parser_object = domain_parser()
        self.train_data_name = train_data_name
        self.rule_calculation = rule_extraction()

        self.time_now = str(datetime.datetime.now())[0:19].replace(" ", "_")

        if algorithm == 'NB':
            self.model = self.create_model_NB()
        elif algorithm == 'RF':
            self.model = self.create_model_RF()

    def __txt_to_list(self, txt_object):

        lst = []

        for line in txt_object:
            lst.append(line.strip())

        txt_object.close()

        return lst

    def preparing_train_data(self, legitimate_file="data_legitimate_36400.json", phishing_file="data_phishing_37175.json"):

        try:
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
            
            # Extract features
            X = self.extract_features(data)
            
            # Prepare target variable
            le = LabelEncoder()
            y = le.fit_transform([item['label'] for item in data])
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            return X_train, y_train, X_test, y_test

        except:
            self.logger.error("Training data preparation failed / Error : {0}".format(format_exc()))
            return None, None, None, None

    def extract_features(self, data):
        features = []
        for item in data:
            url = item['url']
            features.append([
                len(url),  # Length of the URL
                url.count('-'),  # Number of hyphens
                url.count('.'),  # Number of dots
                1 if 'login' in url.lower() or 'secure' in url.lower() else 0  # Presence of 'login' or 'secure'
            ])
        return np.array(features)

    def create_model_NB(self):
        X_train, y_train, _, _ = self.preparing_train_data()
        gnb = GaussianNB()
        model = gnb.fit(X_train, y_train)
        return model

    def create_model_RF(self):
        X_train, y_train, _, _ = self.preparing_train_data()
        clf = RandomForestClassifier(n_estimators=10, random_state=0, verbose=1)
        model = clf.fit(X_train, y_train)
        return model

    def model_run(self, test_features):

        if self.model == 'NB':
            model = self.create_model_NB()
        elif self.model == 'RF':
            model = self.create_model_RF()
        else:
            raise ValueError("Invalid algorithm. Please specify 'NB' or 'RF'.")

        y_pred = model.predict(test_features)
        y_prob = model.predict_proba(test_features)

        return y_pred, y_prob

    def output(self, test_urls):

        try:
            # Prepare test data
            test_data = [{'url': url} for url in test_urls]
            test_features = self.extract_features(test_data)

            # Run the model
            y_pred, y_prob = self.model_run(test_features)

            # Prepare results
            results = []
            le = LabelEncoder()
            le.fit(['legitimate', 'phishing'])

            for i, url in enumerate(test_urls):
                result = {
                    'url': url,
                    'predicted_class': le.inverse_transform([y_pred[i]])[0],
                    'probability_phish': y_prob[i][1] * 100,
                    'probability_legitimate': y_prob[i][0] * 100
                }
                results.append(result)

            # Save results to file
            with open("../output/test-output/results-" + self.time_now + ".txt", "w") as f:
                f.write(json.dumps(results))

            return results

        except:
            self.logger.error("Error during prediction / Error : {0}".format(format_exc()))
            return []

    def accuracy(self, legitimate_file="data_legitimate_36400.json", phishing_file="data_phishing_37175.json"):

        try:
            X_train, y_train, X_test, y_test = self.preparing_train_data(legitimate_file, phishing_file)
            if self.model == 'NB':
                model = self.create_model_NB()
            elif self.model == 'RF':
                model = self.create_model_RF()
            else:
                raise ValueError("Invalid algorithm. Please specify 'NB' or 'RF'.")

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.3f}")

            return accuracy

        except:
            self.logger.error("Error calculating accuracy / Error : {0}".format(format_exc()))
            return 0.0

    def confusion_matrix(self, legitimate_file="data_legitimate_36400.json", phishing_file="data_phishing_37175.json"):

        try:
            X_train, y_train, X_test, y_test = self.preparing_train_data(legitimate_file, phishing_file)

            if self.model == 'NB':
                model = self.create_model_NB()
            elif self.model == 'RF':
                model = self.create_model_RF()
            else:
                raise ValueError("Invalid algorithm. Please specify 'NB' or 'RF'.")

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)

            return cm

        except:
            self.logger.error("Error generating confusion matrix / Error : {0}".format(format_exc()))
            return None

# Example usage
if __name__ == "__main__":
    # Initialize the machine learning algorithm with Naive Bayes
    ml_algorithm = machine_learning_algorithm(algorithm='NB')

    # Prepare your test URLs
    test_urls = [
        "http://example.com",
        "http://phishing-site.com",
        "http://legitimate-site.com"
    ]

    # Run the model on the test URLs
    results = ml_algorithm.output(test_urls)

    # Print the results
    print("Results:")
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Probability of Phishing: {result['probability_phish']}%")
        print(f"Probability of Legitimate: {result['probability_legitimate']}%")
        print("-" * 50)

    # Calculate accuracy
    accuracy = ml_algorithm.accuracy()
    print(f"Accuracy: {accuracy:.3f}")

    # Generate confusion matrix
    cm = ml_algorithm.confusion_matrix()
    print("Confusion Matrix:")
    print(cm)