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
from collections import Counter
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect
import gensim

word2vec_path = "../../Libraries/GoogleNews-vectors-negative300.bin.gz"


matplotlib.use('Agg')


class Globals:
    # number of records to process, change it in case you want to try a smaller portion of data to make a rapid test
    max_record = 10000 * 1000

    # evals number for hyperopt parameter tuning
    max_evals = 100

    # Number of records to be written in the file for manual examination
    sample_test_size = 40
    remove_non_english_records = False

    # our categories and their related words
    classes = ['unisex', 'men', 'women', 'kid', 'baby']
    modelName = "model"
    sampling_info = "Normal"
    plot_show = True
    random_state = 40

    calculate_Precision = True
    calculate_Fscore = False
    calculate_Recall = False
    remove_non_english_records = False

    unlabeled_data = None
    labeled_data = None  # to be used for training
    test_df = None
    valid_df = None

    X_train = None
    X_valid = None
    X_test = None

    y_train = None
    y_valid = None
    y_test = None

    X_train_encoded = None
    X_valid_encoded = None
    X_test_encoded = None
    X_unlabeled_encoded = None
    unlabeled_records_corpus = None

    count_vectorizer = None
    encoding_model = None

    plot_folder = 'figs'
    plot_formats = 'png'
    plot_info = "-{}.{}".format(sampling_info, plot_formats)

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
        # plt.savefig('figs/' + Globals.modelName +" "+ 'confusion_matrix.png')
        plt.savefig(
            "{}/{}-{}-{}".format(Globals.plot_folder, Globals.modelName, 'confusion_matrix', Globals.plot_info))
        plt.show()
        # plt.clf()
        return plt

    @staticmethod
    def eda():
        print("***Exploratory Data Analysis**")
        # let's see the distribution of our classes
        print("classes:", Globals.classes)

        print("Number of records with label:", Globals.labeled_data.shape[0])
        print("Number of records without label:", Globals.unlabeled_data.shape[0])
        print("The size of our features is:", Globals.X_train_encoded.shape)
        Globals.display_embeding(Globals.X_train_encoded, Globals.y_train)
        Globals.plot_class_distribution(Globals.labeled_data, title="Labelled Records Distributions")
        print(Globals.unlabeled_data['class'].value_counts())

    @staticmethod
    def plot_class_distribution(data_frame, groupby_feature='product_type', class_name='class', title=" Distributions",
                                starting_index=0):
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
        plt.savefig(
            "{}/{}-{}-{}".format(Globals.plot_folder, Globals.modelName, title, Globals.plot_info))
        plt.show()

    # Bag of Words Counts
    @staticmethod
    def get_count_vectorizer():

        Globals.count_vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2))
        Globals.X_train_encoded = Globals.count_vectorizer.fit_transform(Globals.X_train)
        Globals.X_valid_encoded = Globals.count_vectorizer.transform(Globals.X_valid)
        Globals.X_test_encoded = Globals.count_vectorizer.transform(Globals.X_test)
        Globals.X_unlabeled_encoded = Globals.count_vectorizer.transform(Globals.unlabeled_records_corpus)
        Globals.encoding_model = 'count_vectorizer'
        print('X_train_encoded size:', Globals.X_train_encoded.shape)
        print('X_valid_encoded size:', Globals.X_valid_encoded.shape)
        print('X_test_encoded  size:', Globals.X_test_encoded .shape)
        print('X_unlabeled_encoded  size', Globals.X_unlabeled_encoded.shape)

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
        plt.savefig(
            "{}/{}-{}-{}".format(Globals.plot_folder, Globals.modelName, 'Histogram', Globals.plot_info))
        plt.show()

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
        if hasattr(myModel, 'coef_'):
            elem = range(myModel.coef_.shape[0])
        else:
            elem = range(myModel.feature_importances_.shape[0])
        for class_index in elem:
            word_importances = []
            if hasattr(myModel, 'coef_'):
                word_importances = [(el, index_to_word[i]) for i, el in enumerate(myModel.coef_[class_index])]
            else:
                word_importances = [(el, index_to_word[i]) for i, el in
                                    enumerate(myModel.feature_importances_[class_index])]
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
        plt.savefig(
            "{}/{}-{}-{}".format(Globals.plot_folder, Globals.modelName, 'Important-Keywords', Globals.plot_info))
        plt.show()

    ## Visualizing the embeddings
    @staticmethod
    def display_embeding(X_train, y_train):
        fig = plt.figure(figsize=(7, 7))
        Globals.plot_LSA(X_train, y_train)
        plt.savefig(
            "{}/{}-{}".format(Globals.plot_folder, 'embeddings', Globals.plot_info))
        plt.show()

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

    # Selects k random records to check their prediction manually
    @staticmethod
    def write_random_records(my_df, category=None, file_name=modelName):

        if category is not None:
            my_df = my_df[my_df['class'] == category]
            file_name = file_name + "_category_"

        random_records = random.sample(my_df.index.to_list(), k=Globals.sample_test_size)
        test = my_df.loc[
            random_records, ['class', 'labels', 'product_type', 'full_store_product_url', 'all_text_original']]

        test.to_csv("../data/validate/test_random_unseen_data_" + file_name + ".csv", index=True)

    @staticmethod
    def check_data_size():
        assert Globals.X_train_encoded.shape[0] == len(Globals.y_train)
        assert Globals.X_valid_encoded.shape[0] == len(Globals.y_valid)
        assert Globals.X_test_encoded.shape[0] == len(Globals.y_test)

    @staticmethod
    def undersample_Miss():
        print('before-under sampled dataset shape %s' % Counter(Globals.y_train))
        nm = NearMiss()
        Globals.X_train_encoded, Globals.y_train = nm.fit_resample(Globals.X_train_encoded, Globals.y_train)
        print('under-sampled dataset shape %s' % Counter(Globals.y_train))
        Globals.sampling_info = "NearMiss-underSampler"

    @staticmethod
    def undersample_random():
        print('before-under sampled dataset shape %s' % Counter(Globals.y_train))
        rus = RandomUnderSampler(random_state=42)
        Globals.X_train_encoded, Globals.y_train = rus.fit_resample(Globals.X_train_encoded, Globals.y_train)
        print('under-sampled dataset shape %s' % Counter(Globals.y_train))
        Globals.sampling_info = "Random-UnderSampler"

    @staticmethod
    def oversample_random():
        print('before-over sampled dataset shape %s' % Counter(Globals.y_train))
        rus = RandomOverSampler(random_state=42)
        Globals.X_train_encoded, Globals.y_train = rus.fit_resample(Globals.X_train_encoded, Globals.y_train)
        print('over-sampled dataset shape %s' % Counter(Globals.y_train))

        Globals.sampling_info = "Random-OverSampler"


    @staticmethod
    def overrsample_SMOTE():
        print('before-over sampled dataset shape %s' % Counter(Globals.y_train))
        sm = SMOTE(random_state=Globals.random_state)
        Globals.X_train_encoded, Globals.y_train = sm.fit_resample(Globals.X_train_encoded, Globals.y_train)
        print('over-sampled dataset shape %s' % Counter(Globals.y_train))
        Globals.sampling_info = "SMOTE-OverSampler"

    @staticmethod
    def plot_prediction_probability(probabilty, title):
        plt.xlabel('Prediction Probability')
        plt.ylabel('Number of records')
        plt.hist(probabilty)
        plt.title(title)
        plt.show()
        plt.savefig(
            "{}/{}-{}-{}".format(Globals.plot_folder, Globals.modelName, title, Globals.plot_info))

    @staticmethod
    def _plot_fig(train_results, valid_results, model_name, title):
        colors = ["red", "blue", "green"]
        xs = np.arange(1, train_results.shape[1] + 1)
        plt.figure()
        legends = []
        for i in range(train_results.shape[0]):
            plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
            plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
            legends.append("train-%d" % (i + 1))
            legends.append("valid-%d" % (i + 1))
        plt.xlabel("Epoch")
        plt.ylabel("Normalized Gini")
        plt.title("%s" % model_name)
        plt.legend(legends)
        plt.savefig(
            "{}/{}-{}-{}".format(Globals.plot_folder, Globals.modelName, title, Globals.plot_info))
        plt.close()

    @staticmethod
    def read_data():

        # Read all the records
        df = pd.DataFrame()
        dbfile = open('../data/labeled/labeled_dataV3-1million', 'rb')
        df = pickle.load(dbfile)
        dbfile.close()

        # Read manually labeled records for evaluation
        inputFile = "../data/test/test_random_unseen_data-.csv"
        test_df = pd.read_csv(inputFile, dtype={"Id": str, "True Label": str})
        test_df = test_df.dropna()
        test_df = test_df.drop(['class', 'labels', 'product_type', 'full_store_product_url', 'all_text_original'],
                               axis=1)
        df['Id'] = df.index.astype(str)
        test_df = pd.merge(test_df, df, on='Id')

        df = df.drop('Id', axis=1)
        test_df['labels'] = test_df["True Label"].apply(Globals.classes.index)
        Globals.test_df = test_df

        # in case we want to try a smaller portion of data
        df = df[0:Globals.max_record]

        if Globals.remove_non_english_records == True:
            df = Globals.remove_non_english(df)

        print("Number of total records:", df.shape[0])
        labeled_data = df[df['class'] != '-1'].copy()
        unlabeled_data = df[df['class'] == '-1'].copy()
        Globals.labeled_data = labeled_data
        Globals.unlabeled_data = unlabeled_data

        # Encode the classes to their index
        labeled_data['labels'] = labeled_data['class'].apply(Globals.classes.index)
        labeled_records_labels = labeled_data["labels"].tolist()


        labeled_records_corpus = labeled_data["all_text"].tolist()
        unlabeled_records_corpus = unlabeled_data["all_text"].tolist()
        Globals.unlabeled_records_corpus = unlabeled_records_corpus

        random_records_test = random.sample(test_df.index.to_list(), k=50)
        random_records_valid = set(test_df.index.to_list()) - set(random_records_test)

        Globals.valid_df = test_df.loc[random_records_valid, :]
        Globals.test_df = test_df.loc[random_records_test, :]

        Globals.y_train = labeled_records_labels
        Globals.y_valid = test_df.loc[random_records_valid, "labels"].tolist()
        Globals.y_test = test_df.loc[random_records_test, "labels"].tolist()

        Globals.X_train = labeled_records_corpus
        Globals.X_valid = test_df.loc[random_records_valid, "all_text"].tolist()
        Globals.X_test = test_df.loc[random_records_test, "all_text"].tolist()

    @staticmethod
    def get_average_word2vec(tokens_list, generate_missing=False, k=300):

        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        Globals.encoding_model = "Word2vec"

        if len(tokens_list) < 1:
            return np.zeros(k)
        if generate_missing:
            vectorized = [word2vec[word] if word in word2vec else np.random.rand(k) for word in tokens_list]
        else:
            vectorized = [word2vec[word] if word in word2vec else np.nan for word in tokens_list]
        #   vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
        length = len(vectorized)
        if length > 0:
            summed = np.sum(vectorized, axis=0)
            averaged = np.divide(summed, length)
        else:
            averaged = 0
        return averaged

    @staticmethod
    def get_word2vec_embeddings( df, tokenized_text, generate_missing=False):
        embeddings = df[tokenized_text].apply(lambda tokens: Globals.get_average_word2vec(tokens,
                                                                                          generate_missing=generate_missing))
        return list(embeddings)

    @staticmethod
    def get_word2vec():
        print("Encoding Train Data ...")
        Globals.X_train_encoded = list( Globals.labeled_data['all_tokens'].apply(
            lambda tokens: Globals.get_average_word2vec(tokens)) )
        print("Encoding validation Data ...")
        Globals.X_valid_encoded = list( Globals.valid_df['all_tokens'].apply(
            lambda tokens: Globals.get_average_word2vec(tokens)))
        print("Encoding test Data ...")
        Globals.X_test_encoded = list( Globals.test_df['all_tokens'].apply(
            lambda tokens: Globals.get_average_word2vec(tokens)))

        # Globals.X_train_encoded = Globals.get_word2vec_embeddings(word2vec, Globals.labeled_data, 'all_tokens')
        # Globals.X_valid_encoded  =  Globals.get_word2vec_embeddings(word2vec, Globals.valid_df, 'all_tokens')
        # Globals.X_test_encoded  =  Globals.get_word2vec_embeddings(word2vec, Globals.test_df, 'all_tokens')

    @staticmethod
    def remove_non_english(df):
        non_english = []
        count = 0
        for row in range(0, df.shape[0]):

            str1 = " "
            text = str1.join(df.loc[row, 'title'])
            text += str1.join(df.loc[row, 'product_type'])
            try:
                if not text:
                    text = df.loc[row, 'all_text']
                if not text:
                    non_english.append(row)

                elif (detect(text) != 'en'):
                    count += 1
                    non_english.append(row)

            except:
                print("This row throws error:", row, text)
        df = df.drop(non_english)
        print(count, "non english records are deleted")
        return df
