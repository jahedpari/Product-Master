from LogisticReg import *
from HelperFunctions import *
from GlobalVariables import *
from sklearn.metrics import confusion_matrix


# Fitting  Random Forest
def RandomForestModel(X_train_cv, y_train, X_test_cv,y_test, count_vectorizer,unlabeled_data ):

    model = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                               multi_class='multinomial', n_jobs=-1, random_state=40)
    model.fit(X_train_cv, y_train)


    print("The size of our features is:", X_train_cv.shape)

    # Fitting logistic regression
    model = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                               multi_class='multinomial', n_jobs=-1, random_state=40)


    # Evaluation
    evaluate(model, X_train_cv, y_train)

    #Predict test records
    model.fit(X_train_cv, y_train)
    y_predicted = model.predict(X_test_cv)

    # Inspection
    # Let's see how confident is our classifier
    _ = cal_probability(model, X_test_cv)
    confusion_matrix(y_test, y_predicted)
    plot_hist(y_predicted)

    # let's see how our  model performs on unseen data
    all_records_corpus = unlabeled_data["all_text"].tolist()
    X_test_cv = count_vectorizer.transform(all_records_corpus)
    y_test_counts = model.predict(X_test_cv)

    # add prediction to the unlabeled_data dataframe
    unlabeled_data['labels'] = y_test_counts
    unlabeled_data['class'] = unlabeled_data['labels'].apply(classes.__getitem__)

    # Add prediction probability to the unlabeled_data dataframe
    find_pred_probability(unlabeled_data, model, X_test_cv)

    # Let's select k random records and check their prediction manually
    choose_random_record(unlabeled_data)
