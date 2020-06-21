from xgboost import XGBClassifier
import xgboost as xgb
from GlobalVariables import *
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from Models import ModelClass


class XGBstModel(ModelClass):

    def __init__(self):
        super().__init__("xgboost")
        self.space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
                      'gamma': hp.uniform('gamma', 1, 9),
                      'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                      'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                      'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                      'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                      'n_estimators': 180,
                      'seed': 0
                      }

    def train(self):
        # range of values to evaluate for hyper-parameters

        trials = Trials()
        best_hyperparams = fmin(fn=self.get_score,
                                space=self.space,
                                algo=tpe.suggest,
                                max_evals=100,
                                )  # trials=trials
        best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
        best_hyperparams['min_child_weight'] = int(best_hyperparams['min_child_weight'])

        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)

        self.model = XGBClassifier(**best_hyperparams)


    def get_score(self, space):
        clf = xgb.XGBClassifier(
            n_estimators=space['n_estimators'],
            max_depth=int(space['max_depth']),
            gamma=space['gamma'],

            min_child_weight=int(space['min_child_weight']),
            colsample_bytree=int(space['colsample_bytree']),
            reg_alpha=int(space['reg_alpha'])
        )

        evaluation = [(Globals.X_train_cv, Globals.y_train), (Globals.X_valid_cv, Globals.y_valid)]

        clf.fit(Globals.X_train_cv, Globals.y_train,
                eval_set=evaluation, eval_metric="merror",
                early_stopping_rounds=10, verbose=False)

        pred = clf.predict(Globals.X_valid_cv)
        accuracy = accuracy_score(Globals.y_valid, pred > 0.5)  # pred>0.5
        print("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK}

    def inspection(self):
        super().inspection()

        # to understand how confident is our classifier, see prediction probability
        print("Prediction Probability for test data")
        self.find_pred_probability(Globals.test_df, Globals.X_test_cv,
                                      "Prediction Probability for test data")
        print("Prediction Probability for unlaballed data")
        self.find_pred_probability(Globals.unlabeled_data, Globals.X_unlabeled_cv,
                                      "Prediction Probability for unlaballed data")
