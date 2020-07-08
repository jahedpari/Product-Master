from sklearn.ensemble import RandomForestClassifier
from Utility import *
from Models import ModelClass
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from matplotlib import pyplot


class RandomForestModel(ModelClass):
    """
    A class to represent a RandomForest Classifier in Sklearn.
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
        super().__init__("RF")

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
            'max_depth': hp.choice('max_depth', [10, 15, 20, 25, 30, None]),
            'max_features': hp.choice('max_features', ['auto', 'log2', None]),
            'n_estimators': hp.choice('n_estimators', range(50, 400, 50)),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'class_weight': hp.choice('criterion', ["balanced", "balanced_subsample"]),
            'class_weight': hp.choice('class_weight', ['balanced']),
            'n_jobs': -1
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
        self.model = RandomForestClassifier(**hyperparams, random_state=Globals.random_state)

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
        clf = RandomForestClassifier(**params)

        clf.fit(Globals.X_train_encoded, Globals.y_train)
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

        # Calculate feature importance
        importances = self.model.feature_importances_

        # Sort feature importance in descending order
        indices = np.argsort(importances)[::-1][0:10]
        print("indices", indices)

        # Rearrange feature names so they match the sorted feature importances
        names = [Globals.X_train_encoded.columns[i] for i in indices]

        # Create plot
        plt.figure()
        plt.title("Feature Importance")
        plt.bar(range(1, 10), importances[indices])
        plt.xticks(range(1, 10), names, rotation=90)
        plt.show()

    def __get_score(self, n_estimators, X_valid, y_valid):
        '''
        Return the average MAE over 3 CV folds of random forest model.
        Keyword argument:
        n_estimators -- the number of trees in the forest
       '''
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

        cv = KFold(shuffle=True, n_splits=3)
        n_scores = cross_val_score(model, X_valid, y_valid, scoring='accuracy', cv=cv, error_score='raise')

        print("n_estimators:", n_estimators, "accuracy", n_scores.mean())
        return n_scores.mean()
