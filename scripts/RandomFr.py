from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from HelperFunctions import *
from GlobalVariables import *
from sklearn.metrics import confusion_matrix


# Fitting  Random Forest
def RandomForestModel(X_train_cv, y_train, X_valid_cv, y_valid, count_vectorizer, unlabeled_data, X_test_cv, y_test):
    # num_features_for_split = sqrt(total_input_features)

    # find the best n_estimators for RF
    results = {}
    for n_estimators in range(50, 500, 150):
        results[n_estimators] = get_score(n_estimators, X_valid_cv, y_valid)
    # plt.plot(list(results.keys()), list(results.values()))
    # plt.show()
    n_estimators_best = max(results, key=results.get)
    print("best n_estimators:", n_estimators_best)

    # Fit and Predict
    model = RandomForestClassifier(n_estimators=n_estimators_best, random_state=0)
    model.fit(X_train_cv, y_train)
    y_predicted = model.predict(X_test_cv)
    get_metrics(y_test, y_predicted)

    # Inspection
    # confusion_matrix(y_test, y_predicted)

    # To see how our model performs on unlabelled data
    all_records_corpus = unlabeled_data["all_text"].tolist()
    X_unlabeled_cv = count_vectorizer.transform(all_records_corpus)
    y_unlabeled_predicted = model.predict(X_unlabeled_cv)
    unlabeled_data['labels'] = y_unlabeled_predicted
    unlabeled_data['class'] = unlabeled_data['labels'].apply(classes.__getitem__)

    # Let's select k random records and check their prediction manually
    choose_random_record(unlabeled_data)


def get_score(n_estimators, X_valid, y_valid):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv = KFold(shuffle=True, n_splits=3)
    n_scores = cross_val_score(model, X_valid, y_valid, scoring='accuracy', cv=cv, error_score='raise')

    print("n_estimators:", n_estimators, "accuracy", n_scores.mean())
    return n_scores.mean()
