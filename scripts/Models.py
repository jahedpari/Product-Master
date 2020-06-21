class ModelClass:

    X_train_cv = None
    y_train = None
    X_valid_cv = None
    y_valid = None
    X_test_cv = None
    y_test = None
    count_vectorizer = None
    unlabeled_data = None


def __init__(self,modelName):
    self.modelName = modelName


# def __init__(self, X_train_cv, y_train, X_valid_cv, y_valid, X_test_cv, y_test, count_vectorizer, unlabeled_data,
#              modelName):
#     self.X_train_cv = X_train_cv
#     self.y_train = y_train
#     self.X_valid_cv = X_valid_cv
#     self.y_valid = y_valid
#     self.X_test_cv = X_test_cv
#     self.y_test = y_test
#     self.count_vectorizer = count_vectorizer
#     self.unlabeled_data = unlabeled_data
#     self.modelName = modelName
