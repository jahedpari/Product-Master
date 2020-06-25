import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from Utility import *

def cv(data):
    count_vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2))
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer


# Read data set
df = pd.DataFrame()
dbfile = open('../data/labeled/labeled_dataV3', 'rb')
df = pickle.load(dbfile)
dbfile.close()

classes = ['unisex', 'men', 'women', 'kid', 'baby']

# Read manually labeled records for evaluation
inputFile = "../data/test/test_random_unseen_data-.csv"
test_df = pd.read_csv(inputFile, dtype={"Id": str, "True Label": str})
test_df = test_df.dropna()
test_df = test_df.drop(['class', 'labels', 'product_type', 'full_store_product_url', 'all_text_original'], axis=1)
df['Id'] = df.index.astype(str)
test_df = pd.merge(test_df, df, on='Id')
df = df.drop('Id', axis=1)

labeled_data = df[df['class'] != '-1'].copy()
unlabeled_data = df[df['class'] == '-1'].copy()

# encode the classes to their index
labeled_data['labels'] = labeled_data['class'].apply(classes.index)
print("classes:", classes)
print("number of records", df.shape[0])

# let's see the distribution of our classes
Globals.plot_class_distribution(df, 'product_type', 'class', starting_index=1)

print("Number of total records:", df.shape[0])
print("Number of records with label:", labeled_data.shape[0])

all_records_corpus = labeled_data["all_text"].tolist()
all_records_labels = labeled_data["labels"].tolist()

print(test_df.columns)
test_x = test_df["all_text"].tolist()
test_y = test_df["True Label"].tolist()

X_train, X_valid, y_train, y_valid = train_test_split(all_records_corpus,
                                                      all_records_labels, test_size=0.2,
                                                      random_state=40)

X_train_cv, count_vectorizer = cv(X_train)
X_valid_cv = count_vectorizer.transform(X_valid)
test_x_cv = count_vectorizer.transform(test_x)
