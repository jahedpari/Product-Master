from KNN import KNNModel
from LogisticReg import LogisticRegModel, random
from RandomFr import RandomForestModel
from XGBst import XGBstModel
from Utility import Globals


random.seed(Globals.random_state)

#Reads all the data file and generates different data frames
Globals.read_data()

#choose one of the following two options
#Globals.get_count_vectorizer()
Globals.get_word2vec()

#Generates some EDA plots
Globals.eda()

#Check Globals forother oversampling and undersampling methods
Globals.oversample_random(),

# performs every steps of the process
def pipline(model):
    model.train()
    model.fit()
    model.predict()
    model.save_model()
    model.get_metrics()
#    model.inspection()


print("**** Logistic Regression ****")
lgReg = LogisticRegModel()
#pipline(lgReg)

print("**** Random Forest ****")
rf = RandomForestModel()
pipline(rf)


print("**** XGBoost ****")
xgbst = XGBstModel()
pipline(xgbst)

print("**** KNN ****")
knn = KNNModel()
#pipline(knn)


print("done!")
