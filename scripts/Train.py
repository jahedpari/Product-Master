import pickle
from sklearn.model_selection import train_test_split
from LogisticReg import *
import pandas as pd
import numpy as np

# load data frame pickled file
from RandomFr import RandomForestModel
from XGBst import XGBstModel

df = pd.DataFrame()
#dbfile = open('../data/labeled/labeled_dataV3', 'rb')
dbfile = open('../data/labeled/labeled_dataV3-1million', 'rb')
df = pickle.load(dbfile)
dbfile.close()
df = df[0:max_record]

#Read manually labeled records for evauation
inputFile="../data/test/test_random_unseen_data-.csv"
test_df = pd.read_csv(inputFile, dtype={ "Id":str,"True Label":str})
test_df=test_df.dropna()
test_df = test_df.drop(['class', 'labels','product_type','full_store_product_url', 'all_text_original'], axis=1)
df['Id'] = df.index.astype(str)
test_df=pd.merge(test_df,df,on='Id')
df=df.drop('Id',axis=1)
test_df['labels']=test_df["True Label"].apply(classes.index)

labeled_data = df[df['class'] != '-1'].copy()
unlabeled_data = df[df['class'] == '-1'].copy()

# encode the classes to their index
labeled_data['labels'] = labeled_data['class'].apply(classes.index)
print("classes:", classes)
print("number of records", df.shape[0])

# let's see the distribution of our classes
plot_class_distribution(df, 'product_type', 'class', starting_index=1)
print("Number of total records:", df.shape[0])
print("Number of records with label:", labeled_data.shape[0])

all_records_corpus = labeled_data["all_text"].tolist()
all_records_labels = labeled_data["labels"].tolist()

random.seed(30)
random_records_test = random.sample(test_df.index.to_list(), k=50)
random_records_valid=set(test_df.index.to_list())- set(random_records_test)

X_test = test_df.loc[random_records_test, "all_text"].tolist()
y_test= test_df.loc[random_records_test, "labels"].tolist()

X_valid = test_df.loc[random_records_valid,"all_text"].tolist()
y_valid= test_df.loc[random_records_valid,"labels"].tolist()


X_train= all_records_corpus
y_train = all_records_labels

#X_train, X_valid, y_train, y_valid = train_test_split(all_records_corpus,
                                                      # all_records_labels, test_size=0.2,
                                                   # random_state=40)

X_train_cv, count_vectorizer = cv(X_train)
X_valid_cv = count_vectorizer.transform(X_valid)
X_test_cv = count_vectorizer.transform(X_test)


# EDA
print("The size of our features is:", X_train_cv.shape)
#display_embeding(X_train_cv, y_train)

print("**** Logistic Regression ****")
modelName = "Logistic_Reg-"
LogisticRegModel(X_train_cv, y_train, X_valid_cv, y_valid, count_vectorizer, unlabeled_data, X_test_cv, y_test)


print("**** Random Forest ****")
modelName = "Random Forest-"
RandomForestModel(X_train_cv, y_train, X_valid_cv, y_valid, count_vectorizer, unlabeled_data, X_test_cv, y_test)

print("**** XGBoost ****")
modelName = "XGBoost-"
XGBstModel(X_train_cv, y_train, X_valid_cv, y_valid, count_vectorizer, unlabeled_data, X_test_cv, y_test)

print("done!")
