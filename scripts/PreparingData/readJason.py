import bigjson
import unicodecsv as csv

# records to read from our jason file
start_record = 0
end_record = start_record + 10

inputFile = '../data/raw/products_sample.json'
outputFile = '../data/cleaned/outputFile3-' + str(start_record) + "-" + str(end_record) + '.csv'

# create a writer  to write output
fwriter = csv.writer(open(outputFile, 'wb'), encoding='UTF-8', errors='ignore')

# the list of features we are interested in
features = ["id", "product_type", "vendor_name", "title", "store_domain", "store_product_brand_domain", "description",
            "primary_price", "full_store_product_url"]
fwriter.writerow(features)
features.remove('id')

with open(inputFile, 'rb') as reader:
    data = bigjson.load(reader)
    for i in range(start_record, end_record):
        record = data[i]
        row = [i + 1]
        for feature in features:
            if feature in record:
                row.append(record[feature])
            else:
                row.append('')
        print(i, ":", row)
        fwriter.writerow(row)

print("done")
