from sklearn.linear_model import LogisticRegression
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from Utility import *
from Models import ModelClass


class LogisticRegModel(ModelClass):
    def __init__(self):
        super().__init__("Logistic Regression")

    def train(self):
        space = {'warm_start': hp.choice('warm_start', [True, False]),
                 'fit_intercept': hp.choice('fit_intercept', [True, False]),
                 'tol': hp.uniform('tol', 0.00001, 0.0001),
                 'solver': hp.choice('solver', ['saga']),
                 'max_iter': hp.choice('max_iter', range(500, 1000,100)),
                 'penalty': hp.choice('penalty',['l1','l2']),
                 'C': hp.uniform('C', 0.001, 5),
                  'n_jobs': hp.choice('n_jobs', [-1]),
                 'class_weight' : hp.choice('class_weight', ['balanced']),
                 }

        trials = Trials()

        best_hyperparams = fmin(fn=self.objective_fnc,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials
                                )


        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)

        hyperparams = space_eval(space, best_hyperparams)
        self.model = LogisticRegression(**hyperparams)


    def objective_fnc(self, params):
        clf = LogisticRegression(**params)
        clf.fit(Globals.X_train_encoded, Globals.y_train)

        pred = clf.predict(Globals.X_valid_encoded)
        accuracy = accuracy_score(Globals.y_valid, pred)  # pred>0.5
        print("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK}

    def inspection(self):
        super().inspection()

        # to understand how confident is our classifier, see prediction probability
        print("Prediction Probability for test data")
        self.find_pred_probability(Globals.test_df, Globals.X_test_encoded,
                                   "Prediction Probability for test data")
        print("Prediction Probability for unlaballed data")
        self.find_pred_probability(Globals.unlabeled_data, Globals.X_unlabeled_encoded,
                                   "Prediction Probability for unlaballed data")
