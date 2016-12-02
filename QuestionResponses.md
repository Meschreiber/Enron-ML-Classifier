1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

>The goal of this project is to create a supervised classifier that takes in financial and e-mail data about Enron employees and classifies them as a person of interest ("POI") or not.  In order to accomplish this, we are supplied with two files: a pickled dataset "final_project_dataset.pkl" which includes 21 features on 146 Enron employees\*, and a hand-written list "poi_names.txt" of POI as researched by one of the instructors of this course, Katie Malone.  There are 35 people on this list, but only 18 of them appear in the dataset.  Effectively this list has been used to create one of the 21 features, `poi` which indicates whether the person is a POI or not.  Thus, the dataset can be used for *supervised* learning since labels (POI or not?) are available to help us train a classifier.  In addition to these two files we have a directory of files, each correspondinig to an e-mail written by an Enron employee.  There are approximately 500,000 emails, which were obtained by the Federal Energy Regulatory Commission during its investigation of Enron's collapse.  This dataset is available to the public at https://www.cs.cmu.edu/~./enron/. In regards to the pickled dataset, some issues arise:

>- 18 is a relatively small number of positives to train on, especially considering there are 35 of them and only approximately half are found in the dataset
>- The dataset is relatively small in general.
>- The dataset is sparse.  All of the features except for `poi` include NaN values. While some of these NaNs surely indicate 0 (e.g. 129 people have NaNs as director fees, since presumably most employees are not directors and therefore did not collect director fees) some of them indicate missing information, (e.g. 51 observations, over 1/3, have NaN as salary, and surely no people at Enron were paid $0.) This ambiguity advises caution when deciding whether to fill in 0s for NaNs and in feature selection. The number of NaNs in each feature is displayed below.

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
    
>\* There are actually ony 144 employee names.  The other two are 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' which were treated as outliers out of hand and removed.  Using the <a href = "http://www.mathwords.com/o/outlier.htm">IQR definition</a> of an outlier, I built on the number of NaNs, and found the number of high and low outliers for each feature, as well as negative values: 

|                        |  High_outliers |  Low_outliers  |  NaNs |  Non_outliers  | Negative values
|:-----------------------|:--------------:|:--------------:|:-----:|:--------------:|:--------------:|
| bonus                  |               10|             0|    63|            71| 0|
| deferral_payments      |                6|             0|   106|            32|1|
| deferred_income        |                0|             5|    96|            43|48|
| director_fees          |                0|             4|   128|            12|0|
| exercised_stock_options|               11|             0|    43|            90|0|
| expenses               |                3|             0|    50|            91|0|
| from_messages          |               17|             0|    58|            69|0|
| from_poi_to_this_person|               11|             0|    58|            75|0|
| from_this_person_to_poi|               13|             0|    58|            73|0|
| loan_advances          |                0|             0|   141|             3|0|
| long_term_incentive    |                7|             0|    79|            58|0|
| other                  |               11|             0|    53|            80|0|
| poi                    |               18|             0|     0|           126|0|
| restricted_stock       |               13|             1|    35|            95|1|
| restricted_stock_deferred|              1|             1|   127|            15|15|
| salary                   |              6|             3|    50|            85|0|
| shared_receipt_with_poi  |              2|            0|    58|            84|0|
| to_messages              |              7|             0|    58|            79|0|
| total_payments           |             10|             0|    21|           113|0|
| total_stock_value        |             21|             0|    19|           104| 1| 

>However, the presence of outliers does not indicate that any given feature should be removed. In fact, many POIs had outlier quantities for certain fields, and this would help our classifier distinguish POIs from non-POIs.  Likewise, a high number of NaNs is not necessarily a reason to remove a feature -- 'director_fees' has the second highest percentage of NaNs -- yet all POIs had NaNs, so this feature may help to identify them. 

>I looked more closely into the negative values -- all of the values in deferred_income and restricted_stock_deferred where negative, which made sense since there are deferred payments and should be negative numbers when factored into total_payments or total_stock_value.  However, it seemed fishy to me that there was just one person each in 'deferall_payments', 'restricted_stock', and 'total_stock_value' that had negative numbers when all other entries were positive.  It came down to the entries for two people, "BELFER ROBERT" and "BHATNAGAR SANJAY" being mis-entered because a dash was missing in some places in the enron financial pdf. I entered those in correctly and turned the negative values positive in the deferred features.  While it is important those are negative in a spreadsheet type of document (where some cells are dependent on the values of other cells) negative numbers can make some processes, like SelectKBest with chi-squared values not possible.  And since I was changing all values from negative to positive, it was not unfairly transforming the data.

﻿2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

>I added the following features based on a human understanding of the terms:
>- Percent salary/total payments
>- Percent bonus/total payments
>- Ratio of salary:bonus
>- Ratio of total stock value:total payments

>I decided not to pursue combined e-mail features since this was already explored in lessons for this unit.  Additionally, by using the <a href = "https://public.tableau.com/profile/diego2420#!/vizhome/Udacity/UdacityDashboard">very awesome Enron Visualizer</a> I decided to create percent excercised stock/total stock value and noticed that deferred_income might be a good predictor of POI.

>My first attempt at feature selection was the first type listed on the <a href = "http://scikit-learn.org/stable/modules/feature_selection.html">Feature Selection documentation </a>.  This type of selection removes features with particularly low variance.  At first pass, threshold = .8 * (1 - .8), only poi was removed, indicated that all instances of this feature are either one or zero (on or off) in more than 80% of the samples.  This was no new information since we know that only 12.5% (18/144) of the dataset are POIs.  Upping the variance to a higher threshold output `['from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi', 'to_messages']`.  This makes sense since all of these are e-mail datapoints and have lower numbers than the financial datapoints, thus the variance will also be smaller.  In order to use this low variance selecter, I decided it was necessary to scale the features using the min_max_scaler.  After this, the VarianceThreshold selector removed `['loan_advances', 'restricted_stock_deferred', 'total_payments']`.  I decided to keep this in mind, but turn my attention to other feature selectors, namely decision-tree based feature selection and SelectKBest. 

>Within Kbest, I examined chi-squared and mutual-information scores. I selected these two because in the [feature selection documentation] (http://scikit-learn.org/stable/modules/feature_selection.html) it notes three scores that are good for classification, rather than regression, and three scores that are good for sparse datasets.  Chi-squared and mutual-information met both criteria.  I initially looked at ordered importances and scores when using my entire dataset, before realizing that feature selection should only be done on training data -- otherwise bias may creep into the model. If that was the case, it was time to finally employ a pipeline. 

﻿3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

> Before using the pipeline, I tried using the following classifiers "out of the box": naive Bayes, SVM, decision trees, k-nearest neigbors, SGD, random forests, and adaboost.  Originally, I tested them with only 10% of the dataset as test data and with accuracy as the evaluation metric. (More on that later.)  This did not allow me to meaningfully differentiate between them, as their accuracy was either one of three values: 12/15, 13/15, or 14/15. So, I upped the train/test ratio to 30%, then eventually 40% and changed my evaluation metrics to precision, recall, and f1 score. The f1 scores as well as the training/prediction/total time are found in the table below.  While the differences in training time here are trivial, one can imagine with a larger dataset they would not be.  And while the time may not be in a linear correlation with the size of the dataset for each of these methods, it is a useful evaluation metric to keep in mind for future projects.

|                        |  f1 | training time (s)  | prediction time (s) |  total time (s)  |
|:-----------------------|:--------------:|:--------------:|:-----:|:--------------:|
| K nearest neighnbors   |  .94|0|.002|.002|
| Random forests      |.86|.043|.007|.050|
| Naive Bayes        |.86|.001|0|.001|
| Adaboost          |.85|.294|.006|.300|
| Support vector machine|.82|.002|.001|.003|
| Decision tree               |.81|.009|0|.009|
| Stochastic gradient descent |.81|.002|0|.002|

The clear winner is K nearest neighbors (with testing set as 30% of the dataset, it's f1 was actually 0.95).  It has the second best time (tied with SGD), but that is not a useful evaluation method here.  I had learned about SGD in another course and was excited that I remembered the best loss function in this case was a logistic regression probabilistic classifier rather than the default hinge.  It didn't do too well though.  Part of me wishes that the algorithm could be part of a pipeline/grid search to find the best one, as I suspect in some cases they evaluate quite similarly.  I also eventually hope to develop a feel for which kinds of algorithms work best with which kinds of datasets.

﻿4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]


﻿5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]


﻿6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

###Final Reflection 
This wasn't required (and so you might be understandably loathe to read it) but I wanted to document my thoughts for my own sake.  I was more familiar with ML than many of the other topics presented in this nanodegree (except the statistics portion) because I had taken Andrew Ng's Coursera course on it.  That lulled me into a false sense of security as I watched the videos and completed the quizzes.  This project woke me up the complexities of ML in practice, and I know this is just a teensy baby view into it.  There's just so much out there!  

In addition to getting a peek at all of the possibilities, doing this project taught me to read the discussions on the forums early on.  I actively decided not to look at them until I ran into a problem I was really struggling with.  I figured what I came up with would be more original and more authentically mine.  While those things might be true, I wasted hours going down wrong paths and searching for answers on StackOverflow, trying to understand sklearn documentation, etc. when there was already so much good advice, suited for people of my experience, posted on the forums.  For the last couple of projects I am definitely turning there after spending no more than a couple of hours working on the project before I start running into major issues.


