from sklearn.ensemble import RandomForestClassifier
from Utility import *
from Models import ModelClass
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from matplotlib import pyplot

class RandomForestModel(ModelClass):

    def __init__(self):
        super().__init__("RF")

    def train(self):

        space = {
        'max_depth': hp.choice('max_depth', [10,15,20,25,30,None]),
        'max_features': hp.choice('max_features', ['auto','log2',None]),
        'n_estimators': hp.choice('n_estimators', range(50, 400,50)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'class_weight':hp.choice('criterion', ["balanced", "balanced_subsample"]),
#        'max_samples':hp.choice('max_samples', range(0.2, 1.0, 0.2)),
        'class_weight': hp.choice('class_weight', ['balanced']),
        'n_jobs':-1
        }

        trials = Trials()
        best_hyperparams = fmin(fn=self.objective_func,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=Globals.max_evals,
                                trials=trials
                                )

        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)

        hyperparams = space_eval(space, best_hyperparams)
        self.model = RandomForestClassifier(**hyperparams, random_state=Globals.random_state)


       # find the best n_estimators for RF
       # results = {}
       # for n_estimators in range(50, 500, 50):
       #     results[n_estimators] = self.get_score(n_estimators, Globals.X_valid_encoded, Globals.y_valid)
       # n_estimators_best = max(results, key=results.get)
       # print("best n_estimators:", n_estimators_best)
       # plt.plot(list(results.keys()), list(results.values()))
       # plt.show()
       #self.model = RandomForestClassifier(n_estimators=n_estimators_best, random_state=Globals.random_state)

    def objective_func(self, params):
        clf = RandomForestClassifier(**params)

        clf.fit(Globals.X_train_encoded, Globals.y_train)
        pred = clf.predict(Globals.X_valid_encoded)
        #score = cross_val_score(clf, X, y).mean()

        score = f1_score(Globals.y_valid, pred, average='weighted')
        print("SCORE:", score)
        return {'loss': -score, 'status': STATUS_OK}





    def get_score(self,n_estimators, X_valid, y_valid):
        """Return the average MAE over 3 CV folds of random forest model.
        Keyword argument:
        n_estimators -- the number of trees in the forest
        """
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)


        cv = KFold(shuffle=True, n_splits=3)
        n_scores = cross_val_score(model, X_valid, y_valid, scoring='accuracy', cv=cv, error_score='raise')

        print("n_estimators:", n_estimators, "accuracy", n_scores.mean())
        return n_scores.mean()



    def inspection(self):
        super().inspection()

        # Calculate feature importances
        importances = self.model.feature_importances_


        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1][0:10]
        print("indices",indices)

        # Rearrange feature names so they match the sorted feature importances
        names = [Globals.X_train_encoded.columns[i] for i in indices]

        # Create plot
        plt.figure()
        plt.title("Feature Importance")
        plt.bar(range(1,10), importances[indices])
        plt.xticks(range(1,10), names, rotation=90)

        # Show plot
        plt.show()