import pickle
from sklearn.model_selection import train_test_split
from LogisticReg import *
import pandas as pd
from RandomFr import RandomForestModel

# load data frame pickled file
from XGBst import XGBstModel

df = pd.DataFrame()
dbfile = open('../data/labeled/labeled_dataV3', 'rb')
df = pickle.load(dbfile)
dbfile.close()
df = df[0:max_record]

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

X_train, X_test, y_train, y_test = train_test_split(all_records_corpus,
                                                    all_records_labels, test_size=0.2,
                                                    random_state=40)

X_train_cv, count_vectorizer = cv(X_train)
X_test_cv = count_vectorizer.transform(X_test)

print(type(y_train[0]))

# EDA
print("The size of our features is:", X_train_cv.shape)
display_embeding(X_train_cv, y_train)

print("**** Logistic Regression ****")
# LogisticRegModel(X_train_cv, y_train, X_test_cv, y_test,count_vectorizer,unlabeled_data)


print("**** Random Forest ****")
# RandomForestModel(X_train_cv, y_train, X_test_cv, y_test,count_vectorizer,unlabeled_data)

print("**** XGBoost ****")
XGBstModel(X_train_cv, y_train, X_test_cv, y_test, count_vectorizer, unlabeled_data)

print("done!")
