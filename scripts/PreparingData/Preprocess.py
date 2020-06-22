import nltk
import pandas as pd
import pickle
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from string import digits

inputFile = "../../data/cleaned/outputFile2-0-1000000.csv"
df = pd.read_csv(inputFile, dtype={"primary_price": "string"})
df = df.drop(['id'], axis=1)
print("size=", df.shape)
#df = df[0:10000]


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


# tokenize, remove stop words and numbers
def standardize_text_v2(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    removals = digits + string.punctuation
    table = str.maketrans('', '', removals)
    words = [w.translate(table) for w in tokens]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    remove_list = [',', '(', ')', "'", '"', ' ', "'s", 'nan', "''", ',', '']
    words = [token for token in words if not token in remove_list]
    words = normalize(words)
    return words


# tokenize, remove stop words and numbers
def standardize_text_v3(text):
    removals = ["http", "https", "www", "/"]
    for item in removals:
        text = text.replace("item", "")
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    return tokens


df['title'] = df['title'].astype(str)
df['product_type'] = df['product_type'].astype(str)
df['description'] = df['description'].astype(str)
df['store_domain'] = df['store_domain'].astype(str)
df['vendor_name'] = df['vendor_name'].astype(str)
df['store_product_brand_domain'] = df['store_product_brand_domain'].astype(str)

df['all_text_original'] = df['product_type'] + " " + df['vendor_name'] + " " + df['title'] + " " + df[
    'store_domain'] + " " + df['store_product_brand_domain']
df['vendor_name_original'] = df['vendor_name'].str.lower()

df['product_type'] = df['product_type'].apply(standardize_text_v2)
df['vendor_name'] = df['vendor_name'].apply(standardize_text_v3)
df['store_domain'] = df['store_domain'].apply(standardize_text_v3)
df['description'] = df['description'].apply(standardize_text_v2)
df['title'] = df['title'].apply(standardize_text_v2)
df['store_product_brand_domain'] = df['store_product_brand_domain'].apply(standardize_text_v2)

# combine all token in all columns into one column
df['all_tokens'] = df.values[:, 0:6].sum(axis=1)
df['all_text'] = df['all_tokens'].apply(lambda x: " ".join(x))


dbfile = open('../../data/processed/processed-1milliom', 'wb')
pickle.dump(df, dbfile)
dbfile.close()
print("preprocess Done")
