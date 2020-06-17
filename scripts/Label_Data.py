import nltk
import pandas as pd
import numpy as np
import re
import codecs
import gc
from nltk.corpus import wordnet
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pivottablejs import pivot_ui

gc.collect()

inputFile = "../data/cleaned/outputFile2-0-500000.csv"
df = pd.read_csv(inputFile, dtype={"primary_price": "string"})
df = df.drop(['id'], axis=1)
df = df[0:100]

# our categories and their related words
classes = ['unisex', 'men', 'women', 'kid', 'baby']
manNet = ['man', 'men', 'male', 'gentleman', 'gent', 'masculine', ' manlike', ' mannish']
womanNet = ['woman', 'women', 'lady', 'female' 'ladies', 'girl', 'feminine', 'unmacho', 'metrosexual']
babyNet = ['baby', 'toddler', 'infant', 'babe', 'bambino', 'infant', 'neonate', 'newborn']
kidNet = ['kid', 'child', 'children', 'child', 'youth', 'joni', 'schoolchild', 'schoolgirl', 'schoolkid', 'junior']
unisexNet = ['unisex', 'androgynous', 'genderless', 'unisexual']


# Helper Functions

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


# Cleaning the dataset


# map NLTK’s POS tags
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Lemmatize Normalization
def normalize(tokens):
    lem = WordNetLemmatizer()
    return [lem.lemmatize(token, pos=get_wordnet_pos(token)) for token in tokens]


# clean up  tokenized  data
def standardize_tokens(tokens):
    return [token.lower() for token in tokens]


# Few regular expressions to clean up  text data
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r".com", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.replace(r"nan", "")
    df[text_field] = df[text_field].str.lower()
    # tokenize text
    df[text_field] = df[text_field].apply(word_tokenize)
    # remove stop words
    stop_words = set(stopwords.words("english"))
    df[text_field] = df[text_field].apply(lambda x: [item for item in x if item not in stop_words])
    # normalize text
    df[text_field] = df[text_field].apply(normalize)


df['title'] = df['title'].astype(str)
df['product_type'] = df['product_type'].astype(str)
df['description'] = df['description'].astype(str)
df['store_domain'] = df['store_domain'].astype(str)
df['vendor_name'] = df['vendor_name'].astype(str)
df['store_product_brand_domain'] = df['store_product_brand_domain'].astype(str)

df['all_text_original'] = df['product_type'] + " " + df['vendor_name'] + " " + df['title'] + " " + df[
    'store_domain'] + " " + df['store_product_brand_domain'] + " " + df['description']
df['vendor_name_original'] = df['vendor_name'].str.lower()

standardize_text(df, 'product_type')
standardize_text(df, 'vendor_name')
standardize_text(df, 'title')
standardize_text(df, 'store_domain')
standardize_text(df, 'description')
standardize_text(df, 'store_product_brand_domain')

df['all_tokens'] = df.values[:, 0:6].sum(axis=1)


###  Frequency of the categories and their synonyms in the product information

# count occurence of keyword in the list
def countFreq(product_info, keywordList):
    total_count = 0
    for item in product_info:
        total_count += keywordList.count(item)
    return total_count


df['unisex'] = df['all_tokens'].apply(countFreq, keywordList=unisexNet)
df['men'] = df['all_tokens'].apply(countFreq, keywordList=manNet)
df['women'] = df['all_tokens'].apply(countFreq, keywordList=womanNet)
df['baby'] = df['all_tokens'].apply(countFreq, keywordList=babyNet)
df['kid'] = df['all_tokens'].apply(countFreq, keywordList=kidNet)

df['class'] = '-1'

print(df.head())


# ## Choose the label with the highest occurance of the keyword
# choose the label with the highest occurance of the keyword 
def findLabel(dataFrame):
    maxCount = max(dataFrame)
    if maxCount > 0:
        maxLabel = dataFrame[dataFrame == maxCount].index[0]
    else:
        maxLabel = '-1'
    return maxLabel


df.loc[:, 'class'] = df.loc[:, classes].apply(findLabel, axis=1)

# ## The distribution of classes including Nan
grouped = df.groupby(['class'])
print(grouped['product_type'].agg(np.size))

plot_class_distribution(df, 'product_type', 'class', starting_index=0)

# ## The distribution of classes excluding Nan
plot_class_distribution(df, 'product_type', 'class', starting_index=1)

# ## Find the most common keywords in product information and label records containg those keywords

# ###  1) Define commmon keywords for each category

unisexProduct = ['electronics', 'phone', 'fruit', 'movie', 'vegetable',
                 'seafood', 'ipad', 'video', 'music', 'book', 'dairy',
                 'egg', 'fridge', 'phone', 'supplement', 'cable',
                 'cookware', 'cook', 'novel', 'bike', 'headphone',
                 'appliance', 'battery', 'vitamin', 'fence', 'garden',
                 'speaker', 'camera', 'kitchen', 'radio', 'backpack'
                                                          'frozen', 'food', 'household', 'safety', 'skate']

womanProduct = ['jewellery', 'pregnancy', 'make up', 'nail polish',
                'eye shadow', 'skirt', 'Manicure', 'Pedicure']

menProduct = ['shave', 'tuxedo', 'tie']

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

# ### 2) Find most commmon words in product information which are not included in the categories lists

# ####   Let's inspect word and vocabulary of our data set
# combine all rows' tokens  into one list
all_words = list([a for b in df['all_tokens'].tolist() for a in b])
all_words = list(filter(lambda a: a not in [',', '(', ')'], all_words))

VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))

# Find the most commmon words in product_type
from nltk.probability import FreqDist

fdist = FreqDist(all_words)
print(fdist)
fdist.most_common(10)

# Find the most frequent words which are not included  categories lists
from nltk.probability import FreqDist

fdist = FreqDist(all_words)


# for word,number in fdist.most_common(30):
#     if word not in all_categories_lists:
#         print (word)


# ### 3) Label records containg common keywords
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
                                                                      args=['all_tokens'])

plot_class_distribution(df, 'product_type', 'class', starting_index=1)
print(df['class'].value_counts())

# # Export Labeled Data
df['all_text'] = df['all_tokens'].apply(lambda x: " ".join(x))
df.to_csv("../data/labeled/labeled_dataV13.csv", index=True)

# ## Labeling more records based on vendor names
# If products from a vendor all belong to one particular category (given that at least 10 products are listed), we can assign that category to other products from the same vendor

homo_brands = {}
labeled_data = df[df['class'] != '-1'].copy()
grouped = labeled_data.groupby(['vendor_name_original'])
for key, group in grouped:
    class_group = grouped.get_group(key).groupby(['class'])
    # print(key, len(class_group ), class_group['class'].count() )
    # if all products belong to one category
    if len(class_group) == 1:
        # If at least 10 products are listed for a company
        if (class_group['class'].count()[0]) > 10:
            homo_brands[key] = list(class_group.groups.keys())[0]

print(homo_brands)

homo_vendor_bool = df['vendor_name_original'].apply(lambda x: x in list(homo_brands.keys()))
not_labled_bool = df['class'] == '-1'

# records which are not labeled yet and belong to homo vendor
homo_notLabeld_bool = np.logical_and(not_labled_bool, homo_vendor_bool)
homo_notLabeld_index = df[homo_notLabeld_bool].index

pd.DataFrame(
    {'homo_vendor': homo_vendor_bool, 'not_labled_bool': not_labled_bool, 'homo_notLabeld_bool': homo_notLabeld_bool})

print("not labeled \n", df[not_labled_bool].index)
print("homo \n", df[homo_vendor_bool].index)
print("homo and notlabeled \n", df[homo_notLabeld_bool].index)

print(df['class'].value_counts())


def get_homo_class(x):
    vendor = x['vendor_name_original']
    # print(vendor, homo_brands[vendor])
    return homo_brands[vendor]


df.loc[homo_notLabeld_index, 'class'] = df.loc[homo_notLabeld_index, :].apply(get_homo_class, axis=1)
print(df['class'].value_counts())

## Export Labeled Data

df['all_text'] = df['all_tokens'].apply(lambda x: " ".join(x))
df.to_csv("../data/labeled/labeled_dataV23.csv", index=True)
print(labeled_data.head())
