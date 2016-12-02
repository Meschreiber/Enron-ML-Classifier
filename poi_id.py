#!/usr/bin/python
from time import time
import sys
import pickle
import pprint
pp = pprint.PrettyPrinter(indent=4)
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = features_list = ['poi','salary', 'to_messages', 'deferral_payments',
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
print "Features: %s" % data_dict['METTS MARK'].keys()
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
print

keys_w_nans = dict((key, 0) for key, value in data_dict['METTS MARK'].iteritems())
keys_w_negs = dict((key, 0) for key, value in data_dict['METTS MARK'].iteritems())

for person in data_dict:
    for key, value in data_dict[person].iteritems():
        if value == "NaN":
            keys_w_nans[key] += 1
        elif value < 0:
            keys_w_negs[key] += 1

print "Number of NaNs:"          
pp.pprint(keys_w_nans)

print
print "Number of Negative Values"
pp.pprint(keys_w_negs) 
print

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


### Task 3: Create new feature(s)
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

# Example starting point. Try investigating other evaluation techniques!

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
nbclf = GaussianNB()
print "Naive Bayes Classifier:"
t0 = time()
nbclf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
nbpred = nbclf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
nb_acc = accuracy_score(labels_test, nbpred)
print "accuracy:", nb_acc
print

from sklearn.svm import SVC
svm_clf = SVC(kernel="rbf", C = 10000)
print "Support Vector Machine"
t0 = time()
svm_clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
svm_pred = svm_clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
svm_acc = float(accuracy_score(labels_test, svm_pred))
print "accuracy: ", svm_acc
print

from sklearn import tree
print "Decision Tree"
split = tree.DecisionTreeClassifier(min_samples_split = 2)
t0 = time()
split.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
split_pred = split.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
acc_split = accuracy_score(labels_test, split_pred)
print "accuracy: ", acc_split
print






### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)