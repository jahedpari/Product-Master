import pickle
import random
import numpy as np
import pandas as pd

from PreparingData.Round2Labelling import labeled_data_index_r1, labeled_data_index_r2, plot_class_distribution

sample_test_size = 40
max_record = 1000000
#  Round 3: Labeling more records based on vendor names

# If products from a vendor all belong to one particular category (given that at least 10 products are listed)
# we can assign that category to other products from the same vendor

dbfile = open('../../data/labeled/labeled_dataV2-1million', 'rb')
df = pickle.load(dbfile)
dbfile.close()
print(df.shape)
df = df[0:max_record]

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


def get_homo_class(x):
    vendor = x['vendor_name_original']
    # print(vendor, homo_brands[vendor])
    return homo_brands[vendor]


df.loc[homo_notLabeld_index, 'class'] = df.loc[homo_notLabeld_index, :].apply(get_homo_class, axis=1)

# keep track of record labeled in round 3
labeled_data_index = df[df['class'] != '-1'].index.to_list()
print(len(labeled_data_index))
labeled_data_index_before = labeled_data_index_r1 + labeled_data_index_r2
labeled_data_index_r3 = [i for i in labeled_data_index if i not in labeled_data_index_before]
print(len(labeled_data_index_r3))

print("Number of records labeled in round 3:", len(labeled_data_index_r3))
print("Number of records not labeled yet:", df.shape[0] - len(labeled_data_index))

plot_class_distribution(df, 'product_type', 'class', starting_index=1)

# Export Labeled Data
df.to_csv("../../data/labeled/labeled_dataV3-1million.csv", index=True)
dbfile = open('../../data/labeled/labeled_dataV3-1million', 'wb')
pickle.dump(df, dbfile)
dbfile.close()

# Choose some random records that are labeled in this round in order to check the labeling performance
random_records = random.sample(labeled_data_index_r3, k=sample_test_size)
test = df.loc[random_records, ['class', 'product_type', 'full_store_product_url', 'all_text_original']]
test.to_csv("../../data/validate/test_random_labeled_data_Round3-1million.csv", index=True)
