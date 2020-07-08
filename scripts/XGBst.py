from xgboost import XGBClassifier
import xgboost as xgb
from Utility import *
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from Models import ModelClass


class XGBstModel(ModelClass):
    """
    A class to represent a  XGBoost Classifier in Sklearn.
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
        super().__init__("xgboost")

    def train(self):
        '''
        to train the model using hyperopt parameter tuning

                 Parameters:
                 ----------
                        None
                 ----------
                        None
         '''

        # range of values to evaluate for hyper-parameters
        space = {'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
                 'gamma': hp.uniform('gamma', 0, 20),
                 # 'eta': hp.uniform('eta', 0.1, 1.0),
                 'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
                 'reg_alpha': hp.quniform('alpha', 0, 200, 10),
                 'reg_lambda': hp.uniform('reg_lambda', 0, 5),
                 'n_estimators': hp.choice('n_estimators', range(50, 500, 50)),
                 'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
                 'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
                 'subsample': hp.uniform('subsample', 0.8, 1),
                 }

        trials = Trials()
        best_hyperparams = fmin(fn=self.__objective_func,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=Globals.max_evals,
                                trials=trials
                                )

        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)

        hyperparams = space_eval(space, best_hyperparams)
        self.model = XGBClassifier(**hyperparams)

    def __objective_func(self, params):
        '''
        objective function to be minimized with hyperopt parameter tuning

                 Parameters:
                 ----------
                        params: set of parameters to be used for the classifier

                 Returns:
                 ----------
                        returns the evaluation based on the argument params
         '''
        clf = XGBClassifier(**params)
        evaluation = [(Globals.X_train_encoded, Globals.y_train), (Globals.X_valid_encoded, Globals.y_valid)]

        clf.fit(Globals.X_train_encoded, Globals.y_train,
                eval_set=evaluation, eval_metric="merror",
                early_stopping_rounds=10, verbose=False)

        pred = clf.predict(Globals.X_valid_encoded)
        score = f1_score(Globals.y_valid, pred, average='weighted')
        print("SCORE:", score)
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

        xgb.plot_importance(self.model)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.show()
