#!/usr/bin/python
import sys
import pickle
import pprint
pp = pprint.PrettyPrinter(indent=4)
import pandas as pd
import numpy as np
from time import time
import operator
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'to_messages', 'deferral_payments',
                 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock',
                 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value',
                 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi',
                 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Explore features of data

print "Number of  people: %d" % (len(data_dict))
print "Number of features: %d" % (len(data_dict['METTS MARK']))
print
poi_names = open("../final_project/poi_names.txt").read().split('\n')
poi_y = [name for name in poi_names if "(y)" in name]
poi_n = [name for name in poi_names if "(n)" in name]
print "Number of POI in poi_names.txt: %d" % len(poi_y + poi_n)

poi_count = 0
for person in data_dict:
    if data_dict[person]["poi"]==1:
        poi_count +=1 
print "Number of POI in dataset: %d" % (poi_count)
print

def neg_nan_count(data_dict):
    '''returns a dictionary containing the number of NaNs for each feature and the number of negative numbers for each feature'''
    keys_w_nans = dict((key, 0) for key, value in data_dict['METTS MARK'].iteritems())
    keys_w_negs = dict((key, 0) for key, value in data_dict['METTS MARK'].iteritems())
    for person in data_dict:
        for key, value in data_dict[person].iteritems():
            if value == "NaN":
                keys_w_nans[key] += 1
            elif value < 0:
                keys_w_negs[key] += 1
    return keys_w_nans, keys_w_negs

keys_w_nans, keys_w_negs = neg_nan_count(data_dict)

print "Number of NaNs:"          
pp.pprint(keys_w_nans)
print
print "Number of Negative Values"
pp.pprint(keys_w_negs) 

### Task 2: Remove outliers/fix misentries/change negative values
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['expenses'] = 3285 
data_dict['BELFER ROBERT']['total_payments'] = 102500
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = "NaN"

data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'

for person in data_dict:
    if data_dict[person]['deferred_income'] < 0 and data_dict[person]['deferred_income'] != "NaN":
        data_dict[person]['deferred_income'] = - data_dict[person]['deferred_income']
        
for person in data_dict:
    if data_dict[person]['restricted_stock_deferred'] < 0 and data_dict[person]['restricted_stock_deferred'] != "NaN":
        data_dict[person]['restricted_stock_deferred'] = - data_dict[person]['restricted_stock_deferred']

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 3: Create new feature(s)

def create_ratio(data_dict, ratio_name, numerator, denominator):
    '''Calcultes the ratio between a given numerator and denominator
    Names the ratio "ratio_name"
    Returns the the updated dictionary with the ratio values'''
    
    for person in data_dict:
        if data_dict[person][numerator] == 'NaN' or data_dict[person][denominator] == 'NaN':
                data_dict[person][ratio_name] = 'NaN'
        else:
            data_dict[person][ratio_name] = float(data_dict[person][numerator])/float(data_dict[person][denominator])
    return data_dict

data_dict = create_ratio(data_dict, 'sal_total', 'salary', 'total_payments')
data_dict = create_ratio(data_dict, 'bon_total', 'bonus', 'total_payments')
data_dict = create_ratio(data_dict, 'sal_bon', 'salary', 'bonus')
data_dict = create_ratio(data_dict, 'stock_pay', 'total_stock_value', 'total_payments')
data_dict = create_ratio(data_dict, 'excer_stock', 'exercised_stock_options', 'total_stock_value')

features_list.append('sal_total')
features_list.append('bon_total')
features_list.append('sal_bon')
features_list.append('stock_pay')
features_list.append('excer_stock')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def run_clf(clf, features_train, features_test, labels_train, labels_test):
    ''' takes a classifier and training and test data
    prints performance time and metrics'''
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    labels_prediction = clf.predict(features_test)
    print "prediction time:", round(time()-t0, 3), "s"
    report = classification_report(labels_test, labels_prediction)
    print report

print "Naive Bayes Classifier:"
nb_clf = GaussianNB()
run_clf(nb_clf, features_train, features_test, labels_train, labels_test)

print "Support Vector Classifier"
svm_clf = SVC(kernel="rbf", C = 10000)
run_clf(svm_clf, features_train, features_test, labels_train, labels_test)

print "Decision Tree"
split = tree.DecisionTreeClassifier(min_samples_split = 10)
run_clf(split, features_train, features_test, labels_train, labels_test)
<<<<<<< HEAD

neigh_clf = KNeighborsClassifier(n_neighbors = 3)
print "K Nearest Neighbors"
run_clf(neigh_clf, features_train, features_test, labels_train, labels_test)

print "K Nearest Neighbors w Scaling"
run_clf(neigh_clf, preprocessing.MinMaxScaler().fit_transform(features_train),
    preprocessing.MinMaxScaler().fit_transform(features_test), labels_train, labels_test)
=======

neigh_clf = KNeighborsClassifier(n_neighbors = 3)
print "K Nearest Neighbors"
run_clf(neigh_clf, features_train, features_test, labels_train, labels_test)

print "K Nearest Neighbors w Scaling"
run_clf(neigh_clf, preprocessing.MinMaxScaler().fit_transform(features_train),
    preprocessing.MinMaxScaler().fit_transform(features_test), labels_train, labels_test)

print "Stochastic Gradient Descent "
sgd_clf = SGDClassifier(loss="log")
run_clf(sgd_clf,(features_train), (features_test), labels_train, labels_test)

print "Stochastic Gradient Descent w scaling"
run_clf(sgd_clf, preprocessing.MinMaxScaler().fit_transform(features_train),
    preprocessing.MinMaxScaler().fit_transform(features_test), labels_train, labels_test)

print "Random Forest"
rando = RandomForestClassifier(n_estimators=10)
run_clf(rando, features_train, features_test, labels_train, labels_test)

print "Adaboost"
ada_clf = AdaBoostClassifier(n_estimators=100)
run_clf(ada_clf, features_train, features_test, labels_train, labels_test)

### Based on out of the box performance, decided to put kNN and adaboost into pipelines

print "kNN pipeline"
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
select = SelectKBest(score_func = chi2, k = 10)
pca = PCA(n_components = 5)
kneighs = KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)
knn_steps = [('scaling', scaler),
        ('feature_selection', select),
        ('reduce_dim', pca),
        ('k_neighbors', kneighs)]

kNN_pipeline = sklearn.pipeline.Pipeline(knn_steps)
kNN_pipeline.fit(features_train, labels_train)
labels_prediction = kNN_pipeline.predict(features_test)
report = classification_report(labels_test, labels_prediction)
print(report)
>>>>>>> 63e50a3562d4c991bb1f9b5beb79c6d59ada0411

print "Stochastic Gradient Descent "
sgd_clf = SGDClassifier(loss="log")
run_clf(sgd_clf,(features_train), (features_test), labels_train, labels_test)

<<<<<<< HEAD
print "Stochastic Gradient Descent w scaling"
run_clf(sgd_clf, preprocessing.MinMaxScaler().fit_transform(features_train),
    preprocessing.MinMaxScaler().fit_transform(features_test), labels_train, labels_test)

print "Random Forest"
rando = RandomForestClassifier(n_estimators=10)
run_clf(rando, features_train, features_test, labels_train, labels_test)

print "Adaboost"
ada_clf = AdaBoostClassifier(n_estimators=100)
run_clf(ada_clf, features_train, features_test, labels_train, labels_test)

### Based on out of the box performance, decided to put kNN and adaboost into pipelines

print "kNN pipeline"
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
select = SelectKBest(score_func = chi2, k = 10)
pca = PCA(n_components = 5)
kneighs = KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)
knn_steps = [('scaling', scaler),
        ('feature_selection', select),
        ('reduce_dim', pca),
        ('k_neighbors', kneighs)]

kNN_pipeline = sklearn.pipeline.Pipeline(knn_steps)
kNN_pipeline.fit(features_train, labels_train)
labels_prediction = kNN_pipeline.predict(features_test)
report = classification_report(labels_test, labels_prediction)
print(report)


print "adaboost pipeline"
ada = AdaBoostClassifier(n_estimators=100)
ada_steps = [('feature_selection', select),
         ('reduce_dim', pca),
        ('adaboost', ada)]

=======
print "adaboost pipeline"
ada = AdaBoostClassifier(n_estimators=100)
ada_steps = [('feature_selection', select),
         ('reduce_dim', pca),
        ('adaboost', ada)]

>>>>>>> 63e50a3562d4c991bb1f9b5beb79c6d59ada0411
ada_pipeline = sklearn.pipeline.Pipeline(ada_steps)
ada_pipeline.fit(features_train, labels_train)
labels_prediction = ada_pipeline.predict(features_test)
report = classification_report(labels_test, labels_prediction)
print(report)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print "kNN parameter search"
knn_parameters = dict(feature_selection__k = [5, 10, 15, 20],
                  feature_selection__score_func = [chi2, mutual_info_classif],
                  reduce_dim__n_components = [1, 2, 3, 4],
                  k_neighbors__n_neighbors = [3, 5, 7, 9],
                  k_neighbors__n_jobs = [-1],
                  k_neighbors__algorithm = ['auto', 'ball_tree', 'kd_tree']
               )
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)


kNN_gs = GridSearchCV(kNN_pipeline, param_grid = knn_parameters, scoring = 'f1_weighted')
t0 = time()
kNN_gs.fit(features, labels)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
labels_predictions = kNN_gs.predict(features)
print "prediction time:", round(time()-t0, 3), "s"
kNN_clf = kNN_gs.best_estimator_
report = classification_report(labels, labels_predictions)
print(report)
print kNN_clf
print 
print

'''
print "adaboost parameter search"
ada_parameters = dict(feature_selection__k = [5, 10, 15, 20],
                  feature_selection__score_func = [chi2, mutual_info_classif],
                  reduce_dim__n_components = [1, 2, 3, 4],
                  adaboost__n_estimators = [50, 75, 100, 200], )

ada_gs = GridSearchCV(ada_pipeline, param_grid = ada_parameters, scoring = 'f1_weighted')
t0 = time()
ada_gs.fit(features, labels)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
labels_predictions = ada_gs.predict(features)
print "prediction time:", round(time()-t0, 3), "s"
ada_clf = ada_gs.best_estimator_
report = classification_report(labels, labels_predictions)
print(report)
print ada_clf
clf = ada_clf
'''

clf = kNN_clf

SKB_k = SelectKBest(score_func = mutual_info_classif, k = 20)
SKB_k.fit_transform(features, labels)   
feature_scores = SKB_k.scores_
features_selected = [features_list[1:][i]for i in SKB_k.get_support(indices=True)]
features_scores_selected=[feature_scores[i]for i in SKB_k.get_support(indices=True)]

print
print 
print 'Selected Features', features_selected
print 'Feature Scores', features_scores_selected

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
test_classifier(clf, my_dataset, features_list, folds = 1000)

