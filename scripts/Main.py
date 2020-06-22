import pickle
from LogisticReg import *
import pandas as pd
from RandomFr import RandomForestModel
from XGBst import XGBstModel


df = pd.DataFrame()
dbfile = open('../data/labeled/labeled_dataV3-1million', 'rb')
df = pickle.load(dbfile)
dbfile.close()

# Read manually labeled records for evaluation
inputFile = "../data/test/test_random_unseen_data-.csv"
test_df = pd.read_csv(inputFile, dtype={"Id": str, "True Label": str})
test_df = test_df.dropna()
test_df = test_df.drop(['class', 'labels', 'product_type', 'full_store_product_url', 'all_text_original'], axis=1)
df['Id'] = df.index.astype(str)
test_df = pd.merge(test_df, df, on='Id')
df = df.drop('Id', axis=1)
test_df['labels'] = test_df["True Label"].apply(Globals.classes.index)
Globals.test_df = test_df


df = df[0:Globals.max_record]

print("Number of total records:", df.shape[0])
labeled_data = df[df['class'] != '-1'].copy()
Globals.labeled_data = labeled_data

unlabeled_data = df[df['class'] == '-1'].copy()
Globals.unlabeled_data = unlabeled_data

# encode the classes to their index
labeled_data['labels'] = labeled_data['class'].apply(Globals.classes.index)
unlabeled_records_corpus = unlabeled_data["all_text"].tolist()

labeled_records_corpus = labeled_data["all_text"].tolist()
labeled_records_labels = labeled_data["labels"].tolist()
X_train = labeled_records_corpus
Globals.y_train = labeled_records_labels

random.seed(30)
random_records_test = random.sample(test_df.index.to_list(), k=50)
random_records_valid = set(test_df.index.to_list()) - set(random_records_test)
Globals.test_df = test_df.loc[random_records_test,:]
Globals.valid_df = test_df.loc[random_records_valid,:]


X_test = test_df.loc[random_records_test, "all_text"].tolist()
Globals.y_test = test_df.loc[random_records_test, "labels"].tolist()

X_valid = test_df.loc[random_records_valid, "all_text"].tolist()
Globals.y_valid = test_df.loc[random_records_valid, "labels"].tolist()

Globals.X_train_cv, Globals.count_vectorizer = Globals.cv(X_train)
Globals.X_valid_cv = Globals.count_vectorizer.transform(X_valid)
Globals.X_test_cv = Globals.count_vectorizer.transform(X_test)
Globals.X_unlabeled_cv = Globals.count_vectorizer.transform(unlabeled_records_corpus)

Globals.check_data_size()

Globals.eda()



Globals.undersample1()


def pipline(model):
    model.train()
    model.fit()
    model.predict()
    model.get_metrics()
    model.inspection()

print("**** Logistic Regression ****")

lgReg = LogisticRegModel()
pipline(lgReg)


print("**** Random Forest ****")
rf = RandomForestModel()
#pipline(rf)
# rf.train()
# rf.fit()
# rf.predict()
# rf.get_metrics()
# rf.inspection()

print("**** XGBoost ****")
xgbst = XGBstModel()
#pipline(xgbst)
# xgbst.train()
# xgbst.fit()
# xgbst.predict()
# xgbst.get_metrics()
# xgbst.inspection()




print("done!")
