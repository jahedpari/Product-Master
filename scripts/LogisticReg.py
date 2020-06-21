from sklearn.linear_model import LogisticRegression
from GlobalVariables import *



def RandomForestModel(ModelClass):
    def __init__(self):
        super().__init__("Logistic Regression")

    def train(self):
        self.model = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                        multi_class='multinomial', n_jobs=-1, random_state=Globals.random_state)

    def inspection(self):
        super().inspection()

        # to understand how confident is our classifier, see prediction probability
        print("Prediction Probability for test data")
        self.find_pred_probability(self, Globals.test_df, Globals.X_test_cv,
                                   "Prediction Probability for test data")
        print("Prediction Probability for unlaballed data")
        self.find_pred_probability(self, Globals.unlabeled_data, Globals.X_unlabeled_cv,
                                   "Prediction Probability for unlaballed data")
