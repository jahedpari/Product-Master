"""
@author: Fatemeh Jahedpari
"""
from sklearn.linear_model import LogisticRegression
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from Utility import *
from Models import ModelClass


class LogisticRegModel(ModelClass):
    """
    A class to represent a LogisticRegModel Classifier in Sklearn.
    Inherited from  ModelClass

    Attributes
    ----------
        None

    Methods
    -------
    train(): to train the model using hyperopt parameter tuning
    inspection (): provides more insight in addition to its parent inspection method
    """

    def __init__(self):
        """
        Calls parents constructor

        Parameters
        ----------
            None
        """
        super().__init__("Logistic Regression")

    def train(self):
        '''
        to train the model using hyperopt parameter tuning

                 Parameters:
                 ----------
                        None
                 ----------
                        None
         '''
        space = {'warm_start': hp.choice('warm_start', [True, False]),
                 'fit_intercept': hp.choice('fit_intercept', [True, False]),
                 'tol': hp.uniform('tol', 0.00001, 0.0001),
                 'solver': hp.choice('solver', ['saga']),
                 'max_iter': hp.choice('max_iter', range(500, 1000, 100)),
                 'penalty': hp.choice('penalty', ['l1', 'l2']),
                 'C': hp.uniform('C', 0.001, 5),
                 'n_jobs': hp.choice('n_jobs', [-1]),
                 'class_weight': hp.choice('class_weight', ['balanced']),
                 }

        trials = Trials()

        best_hyperparams = fmin(fn=self.__objective_fnc,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=Globals.max_evals,
                                trials=trials
                                )

        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)

        hyperparams = space_eval(space, best_hyperparams)
        self.model = LogisticRegression(**hyperparams)

    def  __objective_fnc(self, params):
        '''
        objective function to be minimized with hyperopt parameter tuning

                 Parameters:
                 ----------
                        params: set of parameters to be used for the classifier

                 Returns:
                 ----------
                        returns the evaluation based on the argument params
         '''
        clf = LogisticRegression(**params)
        clf.fit(Globals.X_train_encoded, Globals.y_train)

        pred = clf.predict(Globals.X_valid_encoded)
        score = f1_score(Globals.y_valid, pred, average='weighted')
        # print("SCORE:", score)
        return {'loss': -score, 'status': STATUS_OK}

    def inspection(self):
        '''
         provides more insight in addition to its parent inspection method

                 Parameters:
                 ----------
                        None

                 Returns:
                 ----------
                        None
         '''
        super().inspection()

        # to understand how confident is our classifier, see prediction probability
        print("Prediction Probability for test data")
        self.find_pred_probability(Globals.test_df, Globals.X_test_encoded,
                                   "Prediction Probability for test data")
        print("Prediction Probability for unlaballed data")
        self.find_pred_probability(Globals.unlabeled_data, Globals.X_unlabeled_encoded,
                                   "Prediction Probability for unlaballed data")
