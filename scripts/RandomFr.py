from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from HelperFunctions import *
from GlobalVariables import *
from sklearn.metrics import confusion_matrix
from Models import ModelClass


class RandomForestModel(ModelClass):

    def __init__(self):
        super().__init__("RF")

    def train(self):
        # num_features_for_split = sqrt(total_input_features)

        # find the best n_estimators for RF
        results = {}
        for n_estimators in range(50, 70, 150):
            results[n_estimators] = self.get_score(n_estimators, Globals.X_valid_cv, Globals.y_valid)
        # plt.plot(list(results.keys()), list(results.values()))
        # plt.show()
        n_estimators_best = max(results, key=results.get)
        print("best n_estimators:", n_estimators_best)

        # Fit and Predict
        model = RandomForestClassifier(n_estimators=n_estimators_best, random_state=0)
        self.model = model








    def get_score(self, n_estimators, X_valid, y_valid):
        """Return the average MAE over 3 CV folds of random forest model.

        Keyword argument:
        n_estimators -- the number of trees in the forest
        """
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        cv = KFold(shuffle=True, n_splits=3)
        n_scores = cross_val_score(model, X_valid, y_valid, scoring='accuracy', cv=cv, error_score='raise')

        print("n_estimators:", n_estimators, "accuracy", n_scores.mean())
        return n_scores.mean()
