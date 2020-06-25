from xgboost import XGBClassifier
import xgboost as xgb
from Utility import *
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from Models import ModelClass


class XGBstModel(ModelClass):

    def __init__(self):
        super().__init__("xgboost")


    def train(self):
        # range of values to evaluate for hyper-parameters
        space = {'max_depth': hp.choice("max_depth", range(3, 18)),
                 'gamma': hp.uniform('gamma', 1, 9),
                 'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                 'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                 'min_child_weight': hp.choice('min_child_weight', range(0, 10)),
                 'n_estimators': 180,
                 'seed': 0
                 }

        trials = Trials()
        best_hyperparams = fmin(fn=self.get_score,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=100,
                                )  # trials=trials
       # best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
       # best_hyperparams['min_child_weight'] = int(best_hyperparams['min_child_weight'])

        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)

        hyperparams = space_eval(space, best_hyperparams)
        self.model = XGBClassifier(**hyperparams)


    def get_score(self,params):
        # clf = XGBClassifier(
        #     n_estimators= space['n_estimators'],
        #     max_depth=int( space['max_depth']),
        #     gamma= space['gamma'],
        #
        #     min_child_weight=int( space['min_child_weight']),
        #     colsample_bytree=int( space['colsample_bytree']),
        #     reg_alpha=int(space['reg_alpha'])
        # )

        clf = XGBClassifier(**params)
        evaluation = [(Globals.X_train_encoded, Globals.y_train), (Globals.X_valid_encoded, Globals.y_valid)]

        clf.fit(Globals.X_train_encoded, Globals.y_train,
                eval_set=evaluation, eval_metric="merror",
                early_stopping_rounds=10, verbose=False)

        pred = clf.predict(Globals.X_valid_encoded)
        accuracy = accuracy_score(Globals.y_valid, pred > 0.5)  # pred>0.5
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

        xgb.plot_tree(self.model, num_trees=0)
        plt.rcParams['figure.figsize'] = [50, 10]
        plt.show()

        xgb.plot_importance(self.model)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.show()
