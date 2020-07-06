from KNN import KNNModel
from LogisticReg import LogisticRegModel, random
from RandomFr import RandomForestModel
from XGBst import XGBstModel
from Utility import Globals


random.seed(Globals.random_state)
Globals.read_data()

#choose one of the following options
Globals.get_count_vectorizer()
#Globals.get_word2vec()


Globals.oversample_random(),


#to perfrom EDA
Globals.eda()


def pipline(model):
    model.train()
    model.fit()
    model.predict()
    model.save_model(model_name=Globals.modelName)
    model.get_metrics()
    model.inspection()


print("**** Logistic Regression ****")
lgReg = LogisticRegModel()
#pipline(lgReg)

print("**** Random Forest ****")
rf = RandomForestModel()
#pipline(rf)


print("**** XGBoost ****")
xgbst = XGBstModel()
#pipline(xgbst)

print("**** KNN ****")
knn = KNNModel()
#pipline(knn)


print("done!")
