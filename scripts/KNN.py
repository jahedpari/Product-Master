from sklearn.neighbors import KNeighborsClassifier

from Utility import *
from Models import ModelClass
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval


class KNNModel(ModelClass):
    """
    A class to represent a KNN Classifier in Sklearn.
    Inherited from  ModelClass

    Attributes
    ----------
        None

    Methods
    -------
    train(): to train the model using hyperopt parameter tuning
    """

    def __init__(self):
        """
        Calls parents constructor

        Parameters
        ----------
            None
        """
        super().__init__("KNN")

    def train(self):
        '''
        to train the model using hyperopt parameter tuning

                 Parameters:
                 ----------
                        None
                 ----------
                        None
         '''
        space = {
            'n_neighbors': hp.choice('n_neighbors', range(1, 5000)),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'n_jobs': hp.choice('n_jobs', [-1]),

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
        self.model = KNeighborsClassifier(**hyperparams)

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
        clf = KNeighborsClassifier(**params)
        clf.fit(Globals.X_train_encoded, Globals.y_train)
        pred = clf.predict(Globals.X_valid_encoded)
        score = f1_score(Globals.y_valid, pred, average='weighted')
        print("SCORE:", score)
        return {'loss': -score, 'status': STATUS_OK}
