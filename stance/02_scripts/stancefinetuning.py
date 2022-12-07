import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


# data
print('Importing Data...')
train_path = '~/stance/01_data/traincleaned.csv'
test_path = '~/stance/01_data/testcleaned.csv'


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# Training Data
print('Importing training data...')
tweets_train = train['Tweet'] #specifying training text
y_train = train['Stance'] #speficying training labels/gold

# Testing Data
print('Importing test data...')
tweets_test = test['Tweet'] #specifying testing text
y_test = test['Stance'] #


# Count Vectorizor
#print('Count Vectorizor X and y variables...')
vec = CountVectorizer(analyzer = 'word', ngram_range=(1,3), encoding = 'utf-8') #Transform instances/posts into a matrix of token counts
X_train = vec.fit_transform(tweets_train.values.astype('U')) # uses vectorizer to make matrix of word counts for training data
X_test = vec.transform(tweets_test.values.astype('U')) # uses vectorizer to make matrix of word counts for testing data

# Parameter fine tuning
print('Tuning parameter settings...')
#values for gridsearch
param_grid = {'kernel': ['linear'],'gamma': [0.01, 0.001, 0.0001],'C': [10, 100, 1000]}

clf = GridSearchCV(SVC(), param_grid, scoring='f1_macro', verbose=5)
clf.fit(X_train, y_train)

opt_params = clf.best_params_
print("Best Parameters")
print(opt_params)
#
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
#
grid_results = [opt_params]
#
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    grid_result = [mean, std * 2, params]
    grid_results.append(grid_result)

f = open("gridresults.txt", "w")

for i in grid_results:
    result = str(i)
    f.write(result + '\n')

f.close()
