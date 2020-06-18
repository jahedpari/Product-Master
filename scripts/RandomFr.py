from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from HelperFunctions import *
from GlobalVariables import *
from sklearn.metrics import confusion_matrix


# Fitting  Random Forest
def RandomForestModel(X_train_cv, y_train, X_test_cv, y_test, count_vectorizer, unlabeled_data,test_x_cv, test_y):
    # num_features_for_split = sqrt(total_input_features)


    results = {}
    for n_estimators in range(50, 450, 50):
        results[n_estimators] = get_score( n_estimators, X_train_cv, y_train)
    plt.plot(list(results.keys()), list(results.values()))
    plt.show()
    n_estimators_best = min(results, key=results.get)

    # Predict test records
    model = RandomForestClassifier(n_estimators=n_estimators_best, random_state=0)
    model.fit(X_train_cv, y_train)
    y_predicted = model.predict(test_x_cv)
    get_metrics( test_y, y_predicted)


    # Inspection
    # Let's see how confident is our classifier
    confusion_matrix(test_y, y_predicted)
    plot_hist(y_predicted)




    # let's see how our  model performs on unseen data
    all_records_corpus = unlabeled_data["all_text"].tolist()
    X_unlabeled_cv = count_vectorizer.transform(all_records_corpus)
    y_unlabeled_predicted = model.predict(X_unlabeled_cv)

    # add prediction to the unlabeled_data dataframe
    unlabeled_data['labels'] = y_unlabeled_predicted
    unlabeled_data['class'] = unlabeled_data['labels'].apply(classes.__getitem__)

    # Add prediction probability to the unlabeled_data dataframe
    find_pred_probability(unlabeled_data, model, X_unlabeled_cv)

    # Let's select k random records and check their prediction manually
    choose_random_record(unlabeled_data)


def get_score(n_estimators, X_train_cv, y_train):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

    # Evaluation
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #accuracy,precision,recall,fscore= evaluate(model, X_train_cv, y_train, cv)
    cv = KFold(shuffle=True, n_splits=3)
    n_scores = cross_val_score(model, X_train_cv, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    return n_scores.mean()