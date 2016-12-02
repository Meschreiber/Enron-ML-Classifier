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
    
>\* There are actually ony 144 employee names.  The other two are 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' which were treated as outliers out of hand and removed.  Using the <a href = "http://www.mathwords.com/o/outlier.htm">IQR definition</a> of an outlier, I built on the number of NaNs, and found the number of high and low outliers for each feature: 

>`                           High_outliers  Low_outliers  NaNs  Non_outliers
> bonus                                 10             0    63            71
> deferral_payments                      6             0   106            32
> deferred_income                        0             5    96            43
> director_fees                          0             4   128            12
> exercised_stock_options               11             0    43            90
> expenses                               3             0    50            91
> from_messages                         17             0    58            69
> from_poi_to_this_person               11             0    58            75
> from_this_person_to_poi               13             0    58            73
> loan_advances                          0             0   141             3
> long_term_incentive                    7             0    79            58
> other                                 11             0    53            80
> poi                                   18             0     0           126
> restricted_stock                      13             1    35            95
> restricted_stock_deferred              1             1   127            15
> salary                                 6             3    50            85
> shared_receipt_with_poi                2             0    58            84
> to_messages                            7             0    58            79
> total_payments                        10             0    21           113
> total_stock_value                     21             0    19           104`  

>However, the presence of outliers does not indicate that any given feature should be removed. In fact, many of POIs had outlier quantities for certain fields, and this would help our classifier distinguish POIs from non-POIs.  Likewise, a high number of NaNs is not necessarily a reason to remove a feature -- 'director_fees' has the second highest percentage of NaNs -- yet none of the POIs had NaNs, so this feature too may help to identify them. 

﻿2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

>I added the following features based on a human understanding of the terms:
>- Percent salary/total payments
>- Percent bonus/total payments
>- Ratio of salary:bonus
>- Ratio of total stock value:total payments

>I decided not to pursue combined e-mail features since this was already explored in lessons for this unit.  Additionally, by using the <a href = "https://public.tableau.com/profile/diego2420#!/vizhome/Udacity/UdacityDashboard">very awesome Enron Visualizer</a> I decided to create percent excercised stock/total stock value and noticed that deferred_income might be a good predictor of POI.

>My first attempt at feature selection was the first type listed on the <a href = "http://scikit-learn.org/stable/modules/feature_selection.html">Feature Selection documentation </a>.  This type of selection removes features with particularly low variance.  At first pass, threshold = .8 * (1 - .8), only poi was removed, indicated that all instances of this feature are either one or zero (on or off) in more than 80% of the samples.  This was no new information since we know that only 12.5% (18/144) of the dataset are POIs.  Upping the variance to a higher threshold output `['from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi', 'to_messages']`.  This makes sense since all of these are e-mail datapoints and have lower numbers than the financial datapoints, thus the variance will also be smaller.  In order to use this low variance selecter, I decided it was necessary to scale the features using the min_max_scaler.  After this, the VarianceThreshold selector removed `['loan_advances', 'restricted_stock_deferred', 'total_payments']`.  I decided to keep this in mind, but turn my attention to other feature selectors.

>I used tree-based feature selection and found these feature importances:
>`[('deferred_income', 0.09355854423354773),
> ('bonus', 0.083194102634256487),
> ('salary', 0.05939896193731934),
> ('total_stock_value', 0.056481630853753295),
> ('other', 0.054656707231666789),
> ('long_term_incentive', 0.051880630391807112),
> ('exercised_stock_options', 0.051637191394683965),
> ('from_messages', 0.050745556314803431),
> ('total_payments', 0.046024097164250065),
> ('stock_pay', 0.045476735100144758),
> ('expenses', 0.04384394779105364),
> ('bon_total', 0.041959184371542189),
> ('from_this_person_to_poi', 0.040367553272505526),
> ('to_messages', 0.039731909342403317),
> ('restricted_stock', 0.038377335506411356),
> ('shared_receipt_with_poi', 0.038290522190466844),
> ('sal_bon', 0.037110398228410654),
> ('sal_total', 0.036245668012236491),
> ('excer_stock', 0.034295743514794863),
> ('from_poi_to_this_person', 0.027700376765628899),
> ('deferral_payments', 0.019912228458090802),
> ('restricted_stock_deferred', 0.0050050066514957629),
> ('loan_advances', 0.0035303006608425349),
> ('director_fees', 0.00057566797788414137)]`

>I used K-best and found these feature scores:
>`[('loan_advances', 549702499.04251242),
> ('total_payments', 291743055.52305174),
> ('total_stock_value', 276569697.03888297),
> ('exercised_stock_options', 237947643.76971057),
> ('bonus', 41546794.079927064),
> ('restricted_stock', 37551575.915919423),
> ('deferred_income', 20469996.744445831),
> ('other', 18029146.221745741),
> ('long_term_incentive', 13273623.898099067),
> ('salary', 3463395.416699328),
> ('restricted_stock_deferred', 2918369.7142857141),
> ('deferral_payments', 575829.04306187853),
> ('expenses', 349013.77484335151),
> ('director_fees', 205309.42857142858),
> ('shared_receipt_with_poi', 13704.817630381005),
> ('to_messages', 6833.8754052980303),
> ('from_messages', 955.78800082948601),
> ('from_poi_to_this_person', 738.41454424450308),
> ('from_this_person_to_poi', 620.96027717347522),
> ('stock_pay', 89.214421366573902),
> ('bon_total', 18.308612008608346),
> ('sal_total', 0.74596712680387356),
> ('excer_stock', 0.0098678733905965335),
> ('sal_bon', 1.3332479328604013e-06)]`

>It is interesting to note that my composite features are at the top of neither ranking and are completely at the bottom for in the KBest.  'loan_advances' which was removed because of low standardized variance is near the bottom of the the tree ranking, but at the very top of KBest.  'Loan_advances' is extremely sparse -- there are only 3 employees who received them, one of which is a POI. (What did Ken Lay need 81,525,000 loan for in addition to his outrageous salary, bonus, etc. ??) Because of this, I am electing not to keep it. Another low standardized variance feature 'total_payments' appears at the top of the KBest ranking and near the middle of the decision tree ranking.  The third low variance item 'restricted_stock_deferred' does not rank highly on either list.

3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]


4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]


5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]


6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

###Final Reflection 
This wasn't required (and so you might be understandably loathe to read it) but I wanted to document my thoughts for my own sake.  I was more familiar with ML than many of the other topics presented in this nanodegree (except the statistics portion) because I had taken Andrew Ng's Coursera course on it.  That lulled me into a false sense of security as I watched the videos and completed the quizzes.  This project woke me up the complexities of ML in practice, and I know this is just a teensy baby view into it.  There's just so much out there!  

In addition to getting a peek at all of the possibilities, doing this project taught me to read the discussions on the forums early on.  I actively decided not to look at them until I ran into a problem I was really struggling with.  I figured what I came up with would be more original and more authentically mine.  While those things might be true, I wasted hours going down wrong paths and searching for answers on StackOverflow, trying to understand sklearn documentation, etc. when there was already so much good advice, suited for people of my experience, posted on the forums.  For the last couple of projects I am definitely turning there after spending no more than a couple of hours working on the project before I start running into major issues.


