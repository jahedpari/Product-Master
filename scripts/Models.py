import random

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, make_scorer, \
    plot_confusion_matrix
import matplotlib.patches as mpatches
import itertools
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import cross_val_score, KFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from GlobalVariables import Globals
from HelperFunctions import *


class ModelClass:



    def __init__(self,modelName):
        self.modelName = modelName
        Globals.modelName = modelName
        self.model=None
        self.y_predicted=None

    def fit(self):
        self.model.fit(Globals.X_train_cv, Globals.y_train)

    def predict(self):
        self.y_predicted = self.model.predict(Globals.X_test_cv )
        return self.y_predicted

    def get_metrics(self):
        precision = precision_score(Globals.y_test, self.y_predicted, pos_label=None,
                                    average='weighted')
        print('Precision', precision)
        recall = recall_score(Globals.y_test, self.y_predicted, pos_label=None,
                              average='weighted')
        print('Recall', recall)
        f1 = f1_score(Globals.y_test, self.y_predicted, pos_label=None, average='weighted')
        print('F1', f1)
        accuracy = accuracy_score(Globals.y_test, self.y_predicted)
        print('Accuracy', accuracy)

        return accuracy, precision, recall, f1


    def inspection(self):
        print("***information related to test records**")
        cm = confusion_matrix(Globals.y_test, self.y_predicted)
        Globals.plot_confusion_matrix(cm, Globals.classes,title="Test Records Confusion Matrix")


        if hasattr(self.model,'coef_'):
           Globals.get_most_important_features(Globals.count_vectorizer, self.model, n=5)


        print("***information related to unlabelled records**")
        ModelClass.check_prediction_unlabelled(self)
        Globals.plot_class_distribution(Globals.unlabeled_data,title="Unlabelled Records Distributions")
        print(Globals.unlabeled_data['class'].value_counts())




    def check_prediction_unlabelled(self):
        # To see how our model performs on unlabelled data
        X_unlabeled_cv = Globals.X_unlabeled_cv
        y_unlabeled_predicted = self.model.predict(X_unlabeled_cv)
        Globals.unlabeled_data['labels'] = y_unlabeled_predicted
        Globals.unlabeled_data['class'] = Globals.unlabeled_data['labels'].apply(Globals.classes.__getitem__)

        # Let's select k random records and check their prediction manually
        Globals.choose_random_record(Globals.unlabeled_data)

    def find_pred_probability(self, my_df, X_test):
        my_df['probability'] = Globals.cal_probability(self.model, X_test)

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
