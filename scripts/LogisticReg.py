from sklearn.linear_model import LogisticRegression

from HelperFunctions import *
from GlobalVariables import *
from sklearn.metrics import confusion_matrix


# Fitting logistic regression
def LogisticRegModel(X_train_cv, y_train, X_test_cv, y_test, count_vectorizer, unlabeled_data):
    model = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                               multi_class='multinomial', n_jobs=-1, random_state=40)

    # Evaluation
    cv = KFold(shuffle=True, n_splits=10)
    evaluate(model, X_train_cv, y_train, cv)

    # Predict test records
    model.fit(X_train_cv, y_train)
    y_predicted = model.predict(X_test_cv)

    # Inspection
    # Let's see how confident is our classifier
    _ = cal_probability(model, X_test_cv)
    confusion_matrix(y_test, y_predicted)
    plot_hist(y_predicted)

    # let's see how our model performs on unseen data
    all_records_corpus = unlabeled_data["all_text"].tolist()
    X_unlabeled_cv = count_vectorizer.transform(all_records_corpus)
    y_unlabeled_predicted = model.predict(X_unlabeled_cv)

    # Add prediction to the unlabeled_data data frame
    unlabeled_data['labels'] = y_unlabeled_predicted
    unlabeled_data['class'] = unlabeled_data['labels'].apply(classes.__getitem__)

    # Add prediction probability to the unlabeled_data data frame
    find_pred_probability(unlabeled_data, model, X_unlabeled_cv)

    # Let's select k random records and check their prediction manually
    choose_random_record(unlabeled_data)
