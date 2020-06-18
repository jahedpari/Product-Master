import random

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, make_scorer
import matplotlib.patches as mpatches
import itertools
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import cross_val_score, KFold
from sklearn import preprocessing

from GlobalVariables import *


# Helper Functions
def plot_class_distribution(data_frame, groupby_feature, class_name, starting_index=0):
    print("Class Distributions")
    grouped = data_frame.groupby([class_name])
    values = grouped[groupby_feature].agg(np.size)[starting_index:]
    labels = values.index.tolist()
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values)
    plt.xticks(y_pos, labels)
    plt.xlabel('Product categories')
    plt.ylabel('Number of Products')
    if plot_show:
        plt.show()
    plt.savefig('figs/' + modelName + 'Distribution.png')
    print(data_frame[class_name].value_counts())


# Bag of Words Counts
def cv(data):
    count_vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2))
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer


# Since visualizing data in large dimensions is hard, let's project it down to 2.
def plot_LSA(test_data, test_labels, plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue', 'red', 'yellow', 'green']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        orange_patch = mpatches.Patch(color='orange', label='Unisex')
        blue_patch = mpatches.Patch(color='blue', label='Men')
        red_patch = mpatches.Patch(color='red', label='Women')
        yellow_patch = mpatches.Patch(color='yellow', label='Kids')
        green_patch = mpatches.Patch(color='black', label='Baby')
        plt.legend(handles=[orange_patch, blue_patch, red_patch, yellow_patch, green_patch], prop={'size': 10})


def plot_important_words(top_words, top_scores, label, position):
    y_pos = np.arange(len(top_words))
    plt.subplot(position)
    plt.barh(y_pos, top_scores, align='center', alpha=0.5)
    plt.title(label, fontsize=10)
    plt.yticks(y_pos, top_words, fontsize=8)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=15)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)

    fig = plt.figure(figsize=(7, 7))
    plot = plot_confusion_matrix(cm, classes=classes, normalize=False, title='Confusion matrix')
    if plot_show:
        plt.show()
    plt.savefig('figs/' + modelName + 'confusion_matrix.png')
    print(cm)


def plot_hist(y_predicted, bins='auto'):
    _ = plt.hist(y_predicted, bins='auto')
    plt.title("Histogram with 'auto' bins")
    if plot_show:
        plt.show()
    plt.savefig('figs/' + modelName + 'Histogram.png')


def plot_important_features(count_vectorizer, model):
    # Plot the features our classifier is using to make decisions.
    importance = get_most_important_features(count_vectorizer, model, 15)

    top_scores_unisex = [a[0] for a in importance[0]['tops']]
    top_words_unisex = [a[1] for a in importance[0]['tops']]

    top_scores_men = [a[0] for a in importance[1]['tops']]
    top_words_men = [a[1] for a in importance[1]['tops']]

    top_scores_women = [a[0] for a in importance[2]['tops']]
    top_words_women = [a[1] for a in importance[2]['tops']]

    top_scores_kid = [a[0] for a in importance[3]['tops']]
    top_words_kid = [a[1] for a in importance[3]['tops']]

    top_scores_baby = [a[0] for a in importance[4]['tops']]
    top_words_baby = [a[1] for a in importance[4]['tops']]

    unisex_pairs = [(a, b) for a, b in zip(top_words_unisex, top_scores_unisex)]
    men_pairs = [(a, b) for a, b in zip(top_words_men, top_scores_men)]
    unisex_pairs = [(a, b) for a, b in zip(top_words_unisex, top_scores_unisex)]
    men_pairs = [(a, b) for a, b in zip(top_words_men, top_scores_men)]

    fig = plt.figure(figsize=(10, 10))
    plot_important_words(top_words_unisex, top_scores_unisex, "Unisex", 321)
    plot_important_words(top_words_men, top_scores_men, "Men", 322)
    plot_important_words(top_words_women, top_scores_women, "Women", 323)
    plot_important_words(top_words_kid, top_scores_kid, "Kids", 324)
    plot_important_words(top_words_baby, top_scores_baby, "Baby", 325)

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.suptitle("Important Keywords for the classifier", fontsize=7)
    plt.xlabel('Importance')
    if plot_show:
        plt.show()
    plt.savefig('figs/' + modelName + 'Important-Keywords.png')


## Visualizing the embeddings
def display_embeding(X_train, y_train):
    fig = plt.figure(figsize=(7, 7))
    plot_LSA(X_train, y_train)
    if plot_show:
        plt.show()
    plt.savefig('figs/' + modelName + 'embeddings.png')


# Add prediction probability to the unlabeled_data dataframe
def cal_probability(myModel, X_test):
    allRecords_probabilty = myModel.predict_proba(X_test)
    allRecords_max_probabilty = []
    for i in range(0, allRecords_probabilty.shape[0]):
        probablities = allRecords_probabilty[i]
        prob_index = np.argmax(probablities)
        prob_max = max(probablities)
        allRecords_max_probabilty.append(prob_max)

    fig = plt.figure(figsize=(5, 5))
    plt.xlabel('Prediction Probability')
    plt.ylabel('Number of records')
    plt.hist(allRecords_max_probabilty)
    if plot_show:
        plt.show()
    plt.savefig('figs/' + modelName + 'Probability.png')
    return allRecords_max_probabilty


def find_pred_probability(my_df, model, X_test):
    my_df['probability'] = cal_probability(model, X_test)

    confidence_threshold = 0.8
    high_confidence = my_df[my_df['probability'] >= confidence_threshold]
    high_confidence_size = high_confidence.shape[0]
    low_confidence_size = my_df.shape[0] - high_confidence_size
    print("Number of records predicted with confidence greater than {} is {} out of {}".format(confidence_threshold,
                                                                                               high_confidence_size,
                                                                                               my_df.shape[0]))
    print("Number of records predicted with confidence less than {}  is {} out of {}".format(confidence_threshold,
                                                                                             low_confidence_size,
                                                                                             my_df.shape[0]))


# Let's select k random records and check their prediction manually
def choose_random_record(my_df, file_name=""):
    random_records = random.sample(my_df.index.to_list(), k=sample_test_size)
    random_records = random.sample(my_df.index.to_list(), k=sample_test_size)
    test = my_df.loc[
        random_records, ['class', 'labels', 'product_type', 'full_store_product_url', 'all_text_original']]

    test.to_csv("../data/validate/test_random_unseen_data_"+file_name+".csv", index=True)
    plot_class_distribution(my_df, 'product_type', 'class', starting_index=0)









def evaluate(model, X_train, y_train, cv):
    accuracy= -1
    precision= -1
    recall= -1
    fscore= -1

    scoring_accuracy = make_scorer(accuracy_score)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_accuracy)
    accuracy=scores.mean()
    print('Accuracy Mean',accuracy )

    if calculate_Precision:
        scoring_precision_micro = make_scorer(precision_score, average='micro')
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_precision_micro)
        precision = scores.mean()
        print('Precision Mean', precision )

    if calculate_Recall:
        scoring_recall_score_micro = make_scorer(recall_score, average='micro')
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_recall_score_micro)
        recall=scores.mean()
        print('Recall Mean', recall)

    if calculate_Fscore:
        scoring_f1_score_micro = make_scorer(f1_score, average='micro')
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_f1_score_micro)
        fscore=scores.mean()
        print('F1 Mean',fscore )

    return accuracy,precision,recall,fscore


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    print('Precision', precision)

    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')
    print('Recall', recall)

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    print('F1', f1)

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    print('Accuracy', accuracy)

    return accuracy, precision, recall, f1


# Let's look at the features our classifier is using to make decisions.
def get_most_important_features(vectorizer, myModel, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes = {}
    for class_index in range(myModel.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(myModel.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }
    return classes
