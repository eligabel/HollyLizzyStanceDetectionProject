import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import re


#import cleaned data for POS tagging
print('import data')
filepath = '~/SeminarInCL/Project/HollyLizzyStanceDetectionProject/stance/01_data/traincleaned.csv'
dat=pd.read_csv(filepath)
print(dat.head(n=5))

#convert Tweet column to string for POS tagger
listdat = list(dat['Tweet'])
#stringdat = shhh

#py_txt = stringdat
#covert list to string
newlist = []
#py_token = sent_tokenize (listdat)
for i in listdat:
	py_lword = nltk.word_tokenize (i)
	py_tag = nltk.pos_tag (py_lword)
	newlist.append(py_tag)

#make new column in Data Frame for the POS tags
dat['posTags'] = newlist

print('exporting POS tagged file')
dat.to_csv('~/SeminarInCL/Project/HollyLizzyStanceDetectionProject/stance/01_data/traincleanedPOS.csv', index=False)
