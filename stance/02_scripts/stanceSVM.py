# Xdomain fixing class imbalance across data sets
# Tokenisierung der Germeval-Trainingsdaten von 2018

from sklearn import svm
from sklearn.svm import SVC
#from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pandas as pd
#from scipy import sparse

# data
print('Importing Data...')
train_path = '~/stance/01_data/traincleaned.csv'
test_path = '~/stance/01_data/testcleaned.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print('Making subsets...')
trainsub = train[train['Target'] == 'Atheism']
testath = test[test['Target'] == 'Atheism']
testcli = test[test['Target'] == 'Climate Change is a Real Concern']
testfem = test[test['Target'] == 'Feminist Movement']
testhil = test[test['Target'] == 'Hillary Clinton']
testabo = test[test['Target'] == 'Legalization of Abortion']



# Training Data
print('Importing training data...')
tweets_train = trainsub['Tweet'] #specifying training text
y_train = trainsub['Stance'] #speficying training labels/gold

# Testing Data
print('Importing test data...')
#tweets_test = testsub['Tweet'] #specifying testing text
#y_test = testsub['Stance'] #
#test_multi = test['Tweet']
#y_multi = test['Stance']
testxath = testath['Tweet']
testxcli = testcli['Tweet']
testxfem = testfem['Tweet']
testxhil = testhil['Tweet']
testxabo = testabo['Tweet']
y_ath = testath['Stance']
y_cli = testcli['Stance']
y_fem = testfem['Stance']
y_hil = testhil['Stance']
y_abo = testabo['Stance']


# Count Vectorizor
#print('Count Vectorizor X and y variables...')
vec = CountVectorizer(analyzer = 'word', ngram_range=(1,3), encoding = 'utf-8') #Transform instances/posts into a matrix of token counts
X_train = vec.fit_transform(tweets_train.values.astype('U')) # uses vectorizer to make matrix of word counts for training data
#X_test = vec.transform(tweets_test.values.astype('U')) # uses vectorizer to make matrix of word counts for testing data
#xtest_multi = vec.transform(test_multi.values.astype('U'))
xtest_ath = vec.transform(testxath.values.astype('U'))
xtest_cli = vec.transform(testxcli.values.astype('U'))
xtest_fem = vec.transform(testxfem.values.astype('U'))
xtest_hil = vec.transform(testxhil.values.astype('U'))
xtest_abo = vec.transform(testxabo.values.astype('U'))

#Model Train
print('Training model...')
clf = svm.SVC(C=10, gamma=0.01, kernel='linear') #best parameters using grid search

clf.fit(X_train, y_train) #wir trainieren auf Trainingsdaten

#classification_report
print('Writing class reports...')
#y_true, y_pred = y_test, clf.predict(X_test)
#classrep = classification_report(y_true,y_pred, digits=6)

#ytrue_multi, ypred_multi = y_multi, clf.predict(xtest_multi)
#multirep = classification_report(ytrue_multi,ypred_multi, digits=6)

ytrue_ath, ypred_ath = y_ath, clf.predict(xtest_ath)
athrep = classification_report(ytrue_ath,ypred_ath, digits=6)

ytrue_cli, ypred_cli = y_cli, clf.predict(xtest_cli)
clirep = classification_report(ytrue_cli,ypred_cli, digits=6)

ytrue_fem, ypred_fem = y_fem, clf.predict(xtest_fem)
femrep = classification_report(ytrue_fem,ypred_fem, digits=6)

ytrue_hil, ypred_hil = y_hil, clf.predict(xtest_hil)
hilrep = classification_report(ytrue_hil,ypred_hil, digits=6)

ytrue_abo, ypred_abo = y_abo, clf.predict(xtest_abo)
aborep = classification_report(ytrue_abo,ypred_abo, digits=6)

print('writing classification reports')

# report = open('clixath_classreport.txt', 'w')
# report.write(athrep)

report = open('athxcli_classreport.txt', 'w')
report.write(clirep)

report = open('athxfem_classreport.txt', 'w')
report.write(femrep)

report = open('athxhil_classreport.txt', 'w')
report.write(hilrep)

report = open('athxabo_classreport.txt', 'w')
report.write(aborep)

print('Writing predict files...')

# predfile = open('clixanth_preds.txt', 'w')
#
# for i in ypred_ath:
#     predicted = str(i)
#     predfile.write(predicted + '\n')
#
# predfile.close()


predfile = open('athxcli_preds.txt', 'w')

for i in ypred_cli:
    predicted = str(i)
    predfile.write(predicted + '\n')

predfile.close()

predfile = open('athfem_preds.txt', 'w')

for i in ypred_fem:
    predicted = str(i)
    predfile.write(predicted + '\n')

predfile.close()

predfile = open('athxhil_preds.txt', 'w')

for i in ypred_hil:
    predicted = str(i)
    predfile.write(predicted + '\n')

predfile.close()

predfile = open('athxabo_preds.txt', 'w')

for i in ypred_abo:
    predicted = str(i)
    predfile.write(predicted + '\n')

predfile.close()

# multi_report = open('athxmulti_classreport.txt', 'w')
# multi_report.write(multirep)
#
# print('Writing predict file...')
# predfile = open('athxmulti_preds.txt', 'w')
#
# for i in ypred_multi:
#     predicted = str(i)
#     predfile.write(predicted + '\n')
#
# predfile.close()
