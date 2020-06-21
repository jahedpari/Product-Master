import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
import matplotlib.patches as mpatches
import itertools
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import cross_val_score, KFold
matplotlib.use('Agg')

class Globals:
    # Number of records to be written in the file for manual examination
    sample_test_size = 40
    random_state=40
    # number of records to process
    max_record = 50000 #1000000

    # our categories and their related words
    classes = ['unisex', 'men', 'women', 'kid', 'baby']
    modelName = "model"

    plot_show = True

    calculate_Precision = True
    calculate_Fscore = False
    calculate_Recall = False

    unlabeled_data = None
    labeled_data = None
    test_df = None
    valid_df = None
    X_train_cv = None
    y_train = None
    X_valid_cv = None
    y_valid = None
    X_test_cv = None
    y_test = None
    count_vectorizer = None
    X_unlabeled_cv = None

    @staticmethod
    def plot_confusion_matrix(cm, labels,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.winter):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=12)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, fontsize=10)
        plt.yticks(tick_marks, labels, fontsize=10)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] < thresh else "black", fontsize=15)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=10)
        plt.xlabel('Predicted label', fontsize=10)
        print(cm)

        if Globals.plot_show:
            plt.show()
        plt.savefig('figs/' + Globals.modelName + 'confusion_matrix.png')
        return plt



    @staticmethod
    def eda():
        print("***Exploratory Data Analysis**")
        # let's see the distribution of our classes
        print("classes:", Globals.classes)

        print("Number of records with label:", Globals.labeled_data.shape[0])
        print("Number of records without label:", Globals.unlabeled_data.shape[0])
        print("The size of our features is:", Globals.X_train_cv.shape)
        Globals.display_embeding(Globals.X_train_cv, Globals.y_train)
        Globals.plot_class_distribution(Globals.labeled_data, title="Labelled Records Distributions")
        print(Globals.unlabeled_data['class'].value_counts())

    # Helper Functions
    @staticmethod
    def plot_class_distribution(data_frame, groupby_feature='product_type', class_name='class', title=" Distributions",
                                starting_index=1):
        print("Class Distributions")
        grouped = data_frame.groupby([class_name])
        values = grouped[groupby_feature].agg(np.size)[starting_index:]
        labels = values.index.tolist()
        y_pos = np.arange(len(labels))
        plt.title(title, fontsize=12)
        plt.bar(y_pos, values)
        plt.xticks(y_pos, labels)
        plt.xlabel('Product categories')
        plt.ylabel('Number of Products')

        if Globals.plot_show:
            plt.show()

        plt.savefig('figs/' + Globals.modelName + title + '.png')

    # Bag of Words Counts
    @staticmethod
    def cv(data):
        count_vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2))
        emb = count_vectorizer.fit_transform(data)
        return emb, count_vectorizer

    # Since visualizing data in large dimensions is hard, let's project it down to 2.
    @staticmethod
    def plot_LSA(test_data, test_labels, plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['orange', 'blue', 'red', 'yellow', 'black']
        if plot:
            plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                        cmap=matplotlib.colors.ListedColormap(colors))
            orange_patch = mpatches.Patch(color='orange', label='Unisex')
            blue_patch = mpatches.Patch(color='blue', label='Men')
            red_patch = mpatches.Patch(color='red', label='Women')
            yellow_patch = mpatches.Patch(color='yellow', label='Kids')
            green_patch = mpatches.Patch(color='black', label='Baby')
            plt.legend(handles=[orange_patch, blue_patch, red_patch, yellow_patch, green_patch], prop={'size': 10})

    @staticmethod
    def plot_hist(y_predicted, bins='auto'):
        _ = plt.hist(y_predicted, bins='auto')
        plt.title("Histogram with 'auto' bins")
        if Globals.plot_show:
            plt.show()
        plt.savefig('figs/' + Globals.modelName + 'Histogram.png')

    # Let's look at the features our classifier is using to make decisions.
    @staticmethod
    def plot_important_words(top_words, top_scores, label, position):
        y_pos = np.arange(len(top_words))
        plt.subplot(position)
        plt.barh(y_pos, top_scores, align='center', alpha=0.5)
        plt.title(label, fontsize=10)
        plt.yticks(y_pos, top_words, fontsize=8)

    @staticmethod
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

    @staticmethod
    def plot_important_features(count_vectorizer, model):
        importance = Globals.get_most_important_features(count_vectorizer, model, 15)
        # Plot the features our classifier is using to make decisions.
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
        Globals.plot_important_words(top_words_unisex, top_scores_unisex, "Unisex", 321)
        Globals.plot_important_words(top_words_men, top_scores_men, "Men", 322)
        Globals.plot_important_words(top_words_women, top_scores_women, "Women", 323)
        Globals.plot_important_words(top_words_kid, top_scores_kid, "Kids", 324)
        Globals.plot_important_words(top_words_baby, top_scores_baby, "Baby", 325)

        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        plt.suptitle("Important Keywords for the classifier", fontsize=7)
        plt.xlabel('Importance')

        if Globals.plot_show:
            plt.show()
        plt.savefig('figs/' + Globals.modelName + 'Important-Keywords.png')

    ## Visualizing the embeddings
    @staticmethod
    def display_embeding(X_train, y_train):
        fig = plt.figure(figsize=(7, 7))
        Globals.plot_LSA(X_train, y_train)

        if Globals.plot_show:
            plt.show()
        plt.savefig('figs/' + Globals.modelName + 'embeddings.png')

    @staticmethod
    def evaluate(model, X_train, y_train, cv):
        accuracy = -1
        precision = -1
        recall = -1
        fscore = -1

        scoring_accuracy = make_scorer(accuracy_score)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_accuracy)
        accuracy = scores.mean()
        print('Accuracy Mean', accuracy)

        if Globals.calculate_Precision:
            scoring_precision_micro = make_scorer(precision_score, average='micro')
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_precision_micro)
            precision = scores.mean()
            print('Precision Mean', precision)

        if Globals.calculate_Recall:
            scoring_recall_score_micro = make_scorer(recall_score, average='micro')
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_recall_score_micro)
            recall = scores.mean()
            print('Recall Mean', recall)

        if Globals.calculate_Fscore:
            scoring_f1_score_micro = make_scorer(f1_score, average='micro')
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_f1_score_micro)
            fscore = scores.mean()
            print('F1 Mean', fscore)

        return accuracy, precision, recall, fscore

    # Let's select k random records and check their prediction manually
    @staticmethod
    def choose_random_record(my_df, category=None, file_name=modelName):

        if category is not None:
            my_df = my_df[my_df['class'] == category]
            file_name = file_name + "_category_"

        random_records = random.sample(my_df.index.to_list(), k=Globals.sample_test_size)
        test = my_df.loc[
            random_records, ['class', 'labels', 'product_type', 'full_store_product_url', 'all_text_original']]

        test.to_csv("../data/validate/test_random_unseen_data_" + file_name + ".csv", index=True)

    @staticmethod
    def check_data_size():
        assert Globals.X_train_cv.shape[0] == len(Globals.y_train)
        assert Globals.X_valid_cv.shape[0] == len(Globals.y_valid)
        assert Globals.X_test_cv.shape[0] == len(Globals.y_test)
