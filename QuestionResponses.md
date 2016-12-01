1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

>The goal of this project is to create a supervised classifier that takes in financial and e-mail data about Enron employees and classifies them as a person of interest ("POI") or not.  In order to accomplish this, we are supplied with two files: a pickled dataset "final_project_dataset.pkl" which includes 21 features on 146 Enron employees\*, and a hand-written list "poi_names.txt" of POI as researched by one of the instructors of this course, Katie Malone.  There are 35 people on this list, but only 18 of them appear in the dataset.  Effectively this list has been used to create one of the 21 features, "poi" which indicates whether the person is a POI or not.  Thus, the dataset can be used for *supervised* learning since labels (POI or not?) are available to help us train a classifier.  In addition to these two files we have a directory of files, each correspondinig to an e-mail written by an Enron employee.  There are approximately 500,000 emails, which were obtained by the Federal Energy Regulatory Commission during its investigation of Enron's collapse.  This dataset is available to the public at https://www.cs.cmu.edu/~./enron/. In regards to the pickled dataset, some issues arise:

>- 18 is a relatively small number of positives to train on, especially considering there are 35 of them and only approximately half are found in the dataset
>- All of the features except for poi include NaN values. While some of these NaNs surely indicate 0 (e.g. 129 people have NaNs as director fees, since presumably most employees are not directors and therefore did not collect director fees) some of them indicate missing information, (e.g. 51 observations, over 1/3, have NaN as salary, and surely no people at Enron were paid 0$.) This ambiguity advises caution when deciding whether to fill in 0s for NaNs and in feature selection. The number of NaNs in each feature is displayed below.

>`{   'bonus': 64,
    'deferral_payments': 107,
    'deferred_income': 97,
    'director_fees': 129,
    'email_address': 35,
    'exercised_stock_options': 44,
    'expenses': 51,
    'from_messages': 60,
    'from_poi_to_this_person': 60,
    'from_this_person_to_poi': 60,
    'loan_advances': 142,
    'long_term_incentive': 80,
    'other': 53,
    'poi': 0,
    'restricted_stock': 36,
    'restricted_stock_deferred': 128,
    'salary': 51,
    'shared_receipt_with_poi': 60,
    'to_messages': 60,
    'total_payments': 21,
    'total_stock_value': 20}`
    
>\* There are actually ony 144 employee names.  The other two are 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' which were treated as outliers out of hand and removed.  

2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]
3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]
5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]
6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

