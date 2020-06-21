from xgboost import XGBClassifier
import xgboost as xgb
from HelperFunctions import *
from GlobalVariables import *
from sklearn.metrics import confusion_matrix
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from Models import ModelClass




class XGBstModel(ModelClass):

    def __init__(self):
        ModelClass.__init__("xgboost")







    def train(self):
        #range of values to evaluate for each hyperparameter
        space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
                 'gamma': hp.uniform('gamma', 1, 9),
                 'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                  'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                  'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                 'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                 'n_estimators': 180,
                 'seed': 0
                 }

        trials = Trials()
        best_hyperparams = fmin(fn=ModelClass.objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=100,
                                ) #trials=trials

        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)

        best_hyperparams ['max_depth']= int(best_hyperparams ['max_depth'])
        best_hyperparams['min_child_weight'] = int(best_hyperparams['min_child_weight'])

        model = XGBClassifier(**best_hyperparams)

        # Predict test records
        model.fit(ModelClass.X_train_cv, ModelClass.y_train)
        y_predicted = model.predict(ModelClass.X_test_cv)

        # Evaluation
        Globals.get_metrics(ModelClass.y_test,y_predicted)


        # Inspection
        # Let's see how confident is our classifier
        _ = Globals.cal_probability(model, ModelClass.X_test_cv)
        Globals.confusion_matrix(ModelClass.y_test, y_predicted)
        Globals.plot_hist(y_predicted)



        # let's see how our model performs on unseen data

        X_unlabeled_cv = ModelClass.count_vectorizer.transform(all_records_corpus)
        y_unlabeled_predicted = model.predict(X_unlabeled_cv)

        # Add prediction to the unlabeled_data data frame
        ModelClass.unlabeled_data['labels'] = y_unlabeled_predicted
        ModelClass.unlabeled_data['class'] = ModelClass.unlabeled_data['labels'].apply(classes.__getitem__)

        # Add prediction probability to the unlabeled_data data frame
        Globals.find_pred_probability(ModelClass.unlabeled_data, model, X_unlabeled_cv)

        # Let's select k random records and check their prediction manually
        Globals.choose_random_record(ModelClass.unlabeled_data)

        xgb.plot_importance(model)
        plt.figure(figsize=(16, 12))
        plt.show()



    def objective(self, space):
        clf = xgb.XGBClassifier(
            n_estimators=space['n_estimators'],
            max_depth=int(space['max_depth']),
            gamma=space['gamma'],

            min_child_weight=int(space['min_child_weight']),
            colsample_bytree=int(space['colsample_bytree']),
            reg_alpha=int(space['reg_alpha'])
        )

        evaluation = [(ModelClass.X_train_cv, ModelClass.y_train), (ModelClass.X_valid_cv, ModelClass.y_valid)]


        clf.fit(ModelClass.X_train_cv, ModelClass.y_train,
                eval_set=evaluation, eval_metric="merror",
                early_stopping_rounds=10, verbose=False)

        pred = clf.predict(ModelClass.X_valid_cv)
        accuracy = accuracy_score(ModelClass.y_valid, pred>0.5) # pred>0.5
        print("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK}
