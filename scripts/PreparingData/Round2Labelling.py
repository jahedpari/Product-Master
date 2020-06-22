import nltk
import pandas as pd
from nltk.corpus import wordnet
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
import random
from nltk.probability import FreqDist
import pickle

# Helper Functions

# Number of records to be written in the file for manual examination
sample_test_size = 20
# which feature to focus for labeling 'all_tokens',
imp_feature = 'product_type'
# number of records to process
max_record = 1000000


def plot_class_distribution(data_frame, groupby_feature, class_name, starting_index=0):
    grouped = data_frame.groupby([class_name])
    values = grouped[groupby_feature].agg(np.size)[starting_index:]
    labels = values.index.tolist()
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values)
    plt.xticks(y_pos, labels)
    plt.xlabel('Product categories')
    plt.ylabel('Number of Products')
    plt.show()
    print(data_frame[class_name].value_counts())


# Lemmatize Normalization
def normalize(tokens):
    lem = WordNetLemmatizer()
    return [lem.lemmatize(token, pos=get_wordnet_pos(token)) for token in tokens]


# clean up  tokenized  data
def standardize_tokens(tokens):
    return [token.lower() for token in tokens]


# map NLTK’s POS tags
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# our categories and their related words
classes = ['unisex', 'men', 'women', 'kid', 'baby']
manNet = ['man', 'men', 'male', 'gentleman', 'gent', 'masculine', ' manlike', ' mannish']
womanNet = ['woman', 'women', 'lady', 'female' 'ladies', 'girl', 'feminine', 'unmacho', 'metrosexual']
babyNet = ['baby', 'toddler', 'infant', 'babe', 'bambino', 'infant', 'neonate', 'newborn']
kidNet = ['kid', 'child', 'children', 'child', 'youth', 'joni', 'schoolchild', 'schoolgirl', 'schoolkid', 'junior']
unisexNet = ['unisex', 'androgynous', 'genderless', 'unisexual']
all_Nets_list = [manNet, womanNet, babyNet, kidNet, unisexNet]

dbfile = open('../../data/labeled/labeled_dataV1-1million', 'rb')
df = pickle.load(dbfile)
dbfile.close()
print(df.shape)
# df=df[0:max_record]


#  Find records labeled in  round 1
# keep track of record labeled in this round
labeled_data_index_r1 = df[df['class'] != '-1'].index.to_list()

print("Number of records labeled in round 1:", len(labeled_data_index_r1))
print("Number of records not labeled in round 1:", df.shape[0] - len(labeled_data_index_r1))

# Round 2: Find the most common keywords in product information and label records containg those keywords

# 2.1- Define commmon keywords for each category
# if these keywords are in the product information and the product is not labeled so far, can be used to label the products
unisexProduct = ['electronics', 'phone', 'fruit', 'movie', 'vegetable',
                 'seafood', 'ipad', 'video', 'music', 'book', 'dairy',
                 'egg', 'fridge', 'phone', 'supplement', 'cable',
                 'cookware', 'cook', 'novel', 'bike', 'headphone',
                 'appliance', 'battery', 'vitamin', 'fence', 'garden',
                 'speaker', 'camera', 'kitchen', 'radio', 'backpack'
                                                          'frozen', 'food', 'household', 'safety', 'sex toys', 'skate',
                 'tuna', 'home']

womanProduct = ['jewellery', 'pregnancy', 'make up', 'nail polish',
                'eye shadow', 'skirt', 'Manicure', 'Pedicure', 'jewellery', 'bracelet', 'necklace',
                'earring', 'jewelry', 'lingerie']

menProduct = ['shave', 'tuxedo']

kidProduct = ['school', 'disney', 'spider', 'barbie', 'doll']

babyProduct = ['Pacifier', 'Strollers', 'diapers', 'potty', 'walkers',
               'playmat', 'Car Seat', 'lip liner', 'Babyliss', 'maternity',
               'Teether', 'nursery', 'carrier', 'crib', 'Rattle', 'sleeper']

# lemmitize and standardize the all the categories lists
unisexProduct = normalize(unisexProduct)
unisexProduct = standardize_tokens(unisexProduct)
menProduct = normalize(menProduct)
menProduct = standardize_tokens(menProduct)
womanProduct = normalize(womanProduct)
womanProduct = standardize_tokens(womanProduct)
kidProduct = normalize(kidProduct)
kidProduct = standardize_tokens(kidProduct)
babyProduct = normalize(babyProduct)
babyProduct = standardize_tokens(babyProduct)

all_categories_lists = [unisexProduct, menProduct, womanProduct, kidProduct, babyProduct]
all_keyword_lists = all_Nets_list + all_categories_lists

all_keyword_set = set()
for list_a in all_keyword_lists:
    all_keyword_set.update(set(list_a))

# 2.2-Find most commmon words in product information which are not included in the categories lists
#   Let's inspect word and vocabulary of our data set
# combine all rows' tokens  into one list
all_words = list([a for b in df['product_type'] for a in b])
all_words = list(filter(lambda a: a not in [',', '(', ')', "'", '"', ' ', "'s", 'nan'], all_words))

# Find the most frequent words which are not included  categories lists
fdist = FreqDist(all_words)
for word, number in fdist.most_common(30):
    if word not in all_keyword_set:
        print(word, end=', ')


# 3) Label records containg common keywords
# find the total frequency of a list of keywords in a tokenized list
def count_occurance_keyword(tokenized_list, category_list):
    count = 0
    text_data = ' '.join(tokenized_list) + " "
    for keyword in category_list:
        count = text_data.count(keyword + " ")
    return count


def findLabel_commonKeywords(dataFrame, feature):
    count_unisex = count_occurance_keyword(dataFrame[feature], unisexProduct)
    count_men = count_occurance_keyword(dataFrame[feature], menProduct)
    count_woman = count_occurance_keyword(dataFrame[feature], womanProduct)
    count_kid = count_occurance_keyword(dataFrame[feature], kidProduct)
    count_baby = count_occurance_keyword(dataFrame[feature], babyProduct)

    index = ['unisex', 'men', 'women', 'kid', 'baby']
    counters = [count_unisex, count_men, count_woman, count_kid, count_baby]
    frequency = pd.Series(counters, index=index)

    # find label with maximum frequency
    max_frequency = max(frequency)
    max_label = frequency.idxmax() if max_frequency > 0 else '-1'
    return max_label


# notLabled = df[ df['class'] == '-1' ]
not_labled_index = df['class'] == '-1'
df.loc[not_labled_index, 'class'] = df.loc[not_labled_index, :].apply(findLabel_commonKeywords, axis=1,
                                                                      args=[imp_feature])

#  Find records labeled in  round 2
# keep trackes of record labeled in this round
labeled_data_index = df[df['class'] != '-1'].index.to_list()
print(len(labeled_data_index))
labeled_data_index_r2 = [i for i in labeled_data_index if i not in labeled_data_index_r1]
print(len(labeled_data_index_r2))

print("Number of records labeled in round 2:", len(labeled_data_index_r2))
print("Number of records not labeled yet:", df.shape[0] - len(labeled_data_index))

plot_class_distribution(df, 'product_type', 'class', starting_index=1)

# Export Labeled Data
df.to_csv("../../data/labeled/labeled_dataV2-1million.csv", index=True)
dbfile = open('../../data/labeled/labeled_dataV2-1million', 'wb')
pickle.dump(df, dbfile)
dbfile.close()

# Choose some random records that are labeled in this round in order to check the labeling performance
random_records = random.sample(labeled_data_index_r2, k=sample_test_size)
test = df.loc[random_records, ['class', 'product_type', 'full_store_product_url', 'all_text_original']]
test.to_csv("../../data/validate/test_random_labeled_data_Round2-1million.csv", index=True)
