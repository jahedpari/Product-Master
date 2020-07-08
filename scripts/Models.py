"""
@author: Fatemeh Jahedpari
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from Utility import Globals
import numpy as np
from sklearn import metrics
import pickle


class ModelClass:
    """
    A class to represent a classification models in Sklearn.

    Attributes
    ----------
    modelName : str
        the name of the model for reporting purpose
    model : a Sklearn model object
        the model object
    y_predicted : list
        the latest predicted values
    """

    def __init__(self, modelName):
        """
        Constructs all the necessary attributes for the ModelClass object.

        Parameters
        ----------
            modelName : str
                the model name for reporting purpose
        """
        self.modelName = modelName
        Globals.modelName = modelName
        self.model = None
        self.y_predicted = None

    def fit(self):
        '''
         fits the model based on Globals.X_train and Globals.y_train

                Parameters:
                ----------
                    None

                Returns:
                ----------
                    None
        '''
        self.model.fit(Globals.X_train_encoded, Globals.y_train)

    def predict(self):
        '''
        predicts Globals.X_test_encoded and updates the object predicted values

                Parameters:
                ----------
                    None

                Returns:
                ----------
                        returns the predicted values
        '''
        self.y_predicted = self.model.predict(Globals.X_test_encoded)
        return self.y_predicted

    def get_metrics(self):
        '''
        Evaluates the model based on the test data and predicted values

                 Parameters:
                 ----------
                        None

                 Returns:
                 ----------
                         None
         '''
        accuracy = accuracy_score(Globals.y_test, self.y_predicted)
        print('Accuracy', accuracy)

        if Globals.calculate_Precision:
            precision = precision_score(Globals.y_test, self.y_predicted, pos_label=None,
                                        average='weighted')
            print('Precision', precision)

        if Globals.calculate_Recall:
            recall = recall_score(Globals.y_test, self.y_predicted, pos_label=None,
                                  average='weighted')
            print('Recall', recall)

        if Globals.calculate_Fscore:
            f1 = f1_score(Globals.y_test, self.y_predicted, pos_label=None, average='weighted')
            print('F1', f1)

    def inspection(self):
        '''
        Provides insight for the model based on the the predicted values

                 Parameters:
                 ----------
                        None

                 Returns:
                 ----------
                         None
         '''
        print("***information related to test records**")
        cm = confusion_matrix(Globals.y_test, self.y_predicted)
        Globals.plot_confusion_matrix(cm, Globals.classes, title="Test Records Confusion Matrix")
        print(metrics.classification_report(Globals.y_test, self.y_predicted))

        # shows most important features used by model, if the model supports this functionality
        #if hasattr(self.model, 'coef_') or hasattr(self.model, 'feature_importances_'):
         #   Globals.plot_important_features(Globals.count_vectorizer, self.model)

        print("***information related to unlabelled records**")
        # to see how our model performs on unseen data
        ModelClass.check_prediction_unlabelled(self)
        Globals.plot_class_distribution(Globals.unlabeled_data, title="Unlabelled Records Distributions")
        print(Globals.unlabeled_data['class'].value_counts())

    def check_prediction_unlabelled(self):
        '''
        Provide insight how our model performs on unlabelled data

                 Parameters:
                 ----------
                        None

                 Returns:
                 ----------
                         None
         '''
        X_unlabeled_cv = Globals.X_unlabeled_encoded
        y_unlabeled_predicted = self.model.predict(X_unlabeled_cv)
        Globals.unlabeled_data['labels'] = y_unlabeled_predicted
        Globals.unlabeled_data['class'] = Globals.unlabeled_data['labels'].apply(Globals.classes.__getitem__)

        # Let's select k random records and check their prediction manually
        Globals.write_random_records(Globals.unlabeled_data)

    def find_pred_probability(self, my_df, X_test, title="Prediction Probability"):
        '''
        Provides insight how our model performs on argument X_test and append it to  argument my_df

                 Parameters:
                 ----------
                        X_test: list
                        my_df: datafarme
                        title: str, optional

                 Returns:
                 ----------
                         None
         '''
        my_df['probability'] = self.cal_probability(X_test, title)
        confidence_threshold = 0.8
        high_confidence = my_df[my_df['probability'] >= confidence_threshold]
        high_confidence_size = high_confidence.shape[0]
        low_confidence_size = my_df.shape[0] - high_confidence_size
        print("Number of records predicted with confidence greater than {} is {} out of {}".format(confidence_threshold,
                                                                                                   high_confidence_size,
                                                                                                   my_df.shape[0]))
        print("Number of records predicted with confidence less than {}  is {} out of {}".format(confidence_threshold,
                                                                                                 low_confidence_size,
                                                                                                 my_df.shape[0]))


    def cal_probability(self, X_test, title="Prediction Probability"):
        '''
        Finds the prediction probability for argument X_test to understand how confident is the
        model in its predictions

                 Parameters:
                 ----------
                        X_test: list
                        title: str, optional
                 Returns:
                 ----------
                        returns the maximum probability for each record
         '''
        all_records_probabilty = self.model.predict_proba(X_test)
        all_records_max_probabilty = []
        for i in range(0, all_records_probabilty.shape[0]):
            probabilities = all_records_probabilty[i]
            prob_index = np.argmax(probabilities)
            prob_max = max(probabilities)
            all_records_max_probabilty.append(prob_max)
        Globals.plot_prediction_probability(all_records_max_probabilty, title)
        return all_records_max_probabilty


    def save_model(self):
        '''
        saves the model using pickle

                 Parameters:
                 ----------
                        None

                 Returns:
                 ----------
                        None
         '''
        file_name = self.modelName + "-" + Globals.encoding_model + ".sav"
        pickle.dump(self, open(file_name, 'wb'))

    def load_model(self, file_name="model.sav"):
        '''
        load a model using pickle

                 Parameters:
                 ----------
                        None

                 Returns:
                 ----------
                        None
        '''
        self.model = pickle.load(open(file_name, 'rb'))
