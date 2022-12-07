
import pandas as pd
import numpy as np
import re

#Change file path as needed
print('import data')
filepath = '~/Documents/stance/01_data/testdata-taskA-all-annotations.csv'
dat=pd.read_csv(filepath)
print(dat.head(n=5))

#make separate datasets based on target
print('stripping whitespace')
dat['Tweet'].str.strip()

print('Replace @tags')
dat['Tweet'] = dat['Tweet'].str.replace(r'@\S+', '@tag',regex=True)

print('exporting cleaned file')
dat.to_csv('~/Documents/stance/01_data/testcleaned.csv', index=False)
