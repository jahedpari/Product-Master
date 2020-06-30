from sklearn.ensemble import RandomForestClassifier
from Utility import *
from Models import ModelClass
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

class RandomForestModel(ModelClass):

    def __init__(self):
        super().__init__("RF")

    def train(self):

        space = {
        'max_depth': hp.choice('max_depth', range(1, 40,2)),
       # 'max_features': hp.choice('max_features', range(1, 50,5)),
        'max_features': hp.choice('max_features', ['auto','log2',None]),
        'n_estimators': hp.choice('n_estimators', range(10, 400,20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        }

        trials = Trials()

        best_hyperparams = fmin(fn=self.get_score,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials
                                )


        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)

        hyperparams = space_eval(space, best_hyperparams)
        self.model = RandomForestClassifier(**hyperparams, random_state=Globals.random_state)

    def get_score(self, params):
        clf = RandomForestClassifier(**params)

        clf.fit(Globals.X_train_encoded, Globals.y_train)

        pred = clf.predict(Globals.X_valid_encoded)
        accuracy = accuracy_score(Globals.y_valid, pred)  # pred>0.5
        print("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK}
