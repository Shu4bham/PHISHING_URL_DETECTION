
import json
import datetime
import numpy as np

from io import StringIO
from scipy.io import arff
from traceback import format_exc
from domain_parser import domain_parser
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from ns_log import NsLog
from json2arff import json2arff
from rule_extraction import rule_extraction


class machine_learning_algorithm():

    def __init__(self, algorithm, train_data_name="arff_2025-03-20_11_08_45.txt"):

        self.logger = NsLog("log")

        self.path_output_arff = "../output/arff/"
        self.path_test_output = ""

        self.json2arff_object = json2arff()
        self.parser_object = domain_parser()
        self.train_data_name = "arff_2025-03-20_11_08_45.txt" # Replace with your actual filename
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

    def preparing_train_data(self, file_name="arff_2025-03-20_11_08_45.txt"):

        train = []
        target = []

        try:
            train_dataset, train_meta = arff.loadarff(open("{0}{1}".format(self.path_output_arff, file_name), "r"))

            train = train_dataset[train_meta.names()[:-1]]  # everything but the last column
            target = train_dataset[train_meta.names()[len(train_meta.names()) - 1]]  # last column

            train = np.asarray(train.tolist(), dtype=np.float32)  # olay burda
        except:
            self.logger.debug(file_name+"test 65 in algorithm.py")
            self.logger.error("Error : {0}".format(format_exc()))

        return train, target

    def preparing_test_data(self, test_dataset_list):

        try:
            feat_json = open("../output/test-output/json-"+self.time_now+".txt", "w")
            feat_arff = open("../output/test-output/arff-"+self.time_now+".arff", "w")

            "domain_parsed to json without class"
            self.test_parsed_domains = self.parser_object.parse_nonlabeled_samples(test_dataset_list)

            "rule calculation for test samples without class information -- output json format"
            test_features = self.rule_calculation.extraction(self.test_parsed_domains)

            arff_test_str = self.json2arff_object.convert_for_test(test_features, '')

           # feat_json.write(json.dumps(test_features))
            feat_arff.write(arff_test_str)

            feat_arff.close()
            feat_json.close()

            arff_raw = StringIO(arff_test_str)

            test_dataset, test_meta = arff.loadarff(arff_raw)

            test = test_dataset[test_meta.names()]
            test = np.asarray(test.tolist(), dtype=np.float32)
        except:
            self.logger.error("Test verisi ayarlanırken hata  /  Error : {0}".format(format_exc()))

        return test, self.test_parsed_domains

    def create_model_NB(self):

        train, target = self.preparing_train_data()
        gnb = GaussianNB()
        model = gnb.fit(train, target)

        return model

    def create_model_RF(self):
        train, target = self.preparing_train_data()
        clf = RandomForestClassifier(n_estimators=10, random_state=0, verbose=1)
        model = clf.fit(train, target)

        return model

    def model_run(self, test):

        model = self.create_model_RF()

        model_pre = model.predict(test)
        model_probability = model.predict_proba(test)

        model_pre_list = []
        for p in model_pre:
            model_pre_list.append(str(p).replace("b'", "").replace("'", ""))

        model_probability = model_probability.tolist()

        return model_pre_list, model_probability

    def output(self, test_data):

        test, test_parsed_domains = self.preparing_test_data(test_data)
        model_pre, model_probability = self.model_run(test)

        test_parsed_domain = self.test_parsed_domains
        result_list = []

        for test_domain in test_parsed_domain:
            result = {}
            result['domain'] = test_domain['url']
            result['id'] = test_domain['id']
            result['predicted_class'] = model_pre[test_domain['id']]
            result['probability_phish'] = (model_probability[test_domain['id']][1] / sum(model_probability[test_domain['id']])) * 100
            result['probability_legitimate'] = (model_probability[test_domain['id']][0] / sum(model_probability[test_domain['id']])) * 100
            result_list.append(result)

        test_result = open("../output/test-output/result-"+self.time_now+".txt", "w")
        test_result.write(json.dumps(result_list))
        test_result.close()

        return result_list

    def accuracy(self):
        model = self.model
        test_data, test_label = self.preparing_train_data()
        scores = cross_val_score(model, test_data, test_label, cv=10)
        return scores

    # def confusion_matrix(self, name):
    #     test, test_label = self.preparing_train_data(file_name=name)
    #     model_pre, model_pro = self.model_run(test)

    #     test_label_unicode = []

    #     for t in test_label:
    #         test_label_unicode.append(str(t, 'utf-8'))

    #     return confusion_matrix(test_label_unicode, model_pre, labels=['phish', 'legitimate'])
    def confusion_matrix(self, name):
        test, test_label = self.preparing_train_data(file_name=name)
        # model_pre, model_pro = self.model_run(test) # <--- REMOVE THIS LINE
        model_pre = self.model.predict(test) # <--- USE THE EXISTING self.model
        # model_probability = self.model.predict_proba(test) # <-- PROBABILITIES NOT NEEDED FOR CONFUSION MATRIX

        model_pre_list = []
        for p in model_pre:
            model_pre_list.append(str(p).replace("b'", "").replace("'", ""))

        test_label_unicode = []
        for t in test_label:
            test_label_unicode.append(str(t, 'utf-8'))

        return confusion_matrix(test_label_unicode, model_pre_list, labels=['phish', 'legitimate'])





# ... (rest of your algorithm.py code) ...

if __name__ == "__main__":
    # Example evaluation code (similar to evaluate.py)
    ml_algo_rf = machine_learning_algorithm('RF')
    ml_algo_nb = machine_learning_algorithm('NB')

    print("Evaluating Random Forest Model (from algorithm.py):")
    accuracy_scores_rf = ml_algo_rf.accuracy()
    print("  Accuracy Scores (Cross-Validation):", accuracy_scores_rf)
    print("  Average Accuracy (RF):", sum(accuracy_scores_rf) / len(accuracy_scores_rf))
    conf_matrix_rf = ml_algo_rf.confusion_matrix(name="arff_2025-03-20_11_08_45.txt") # Replace arff_*.txt with your ARFF filename
    print("\n  Confusion Matrix (Random Forest):\n", conf_matrix_rf)


    print("\n\nEvaluating Naive Bayes Model (from algorithm.py):")
    accuracy_scores_nb = ml_algo_nb.accuracy()
    print("  Accuracy Scores (Cross-Validation):", accuracy_scores_nb)
    print("  Average Accuracy (NB):", sum(accuracy_scores_nb) / len(accuracy_scores_nb))
    conf_matrix_nb = ml_algo_nb.confusion_matrix(name="arff_2025-03-20_11_08_45.txt") # Replace arff_*.txt with your ARFF filename
    print("\n  Confusion Matrix (Naive Bayes):\n", conf_matrix_nb)