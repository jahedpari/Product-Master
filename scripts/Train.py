import pandas as pd

input_file='..\data\\'+'Labeled-50000.csv'
data=pd.read_csv(+input_file, sep='\t')


print(data)