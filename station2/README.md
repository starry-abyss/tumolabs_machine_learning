# Pre-interview assessment

The project is aimed to predict the HR decision regarding offering a new candidate a company position based on previously accumulated statistics.

The Python notebook can be viewed right in this Github repository. I've used anaconda.com/app to create it, but probably it will work in other flavors of Python notebooks too.

In this project I try several models from scikit-learn: 4 Naive Bayes models, 2 Logistic Regression models, and one K-nearest-neighbors.

All is done using the standard numpy+matplotlib+pandas stack.

Based on plots of different features, this company prefers candidates who are not married, and are OK with business trips and overtime (perhaps, business trips are an essential part of the job).

Education and age also are an important factor for the decision.

Since OverTime, Gender, BusinessTravel, and MaritalStatus are given as strings, I'm using OrdinalEncoder to convert them to numbers.

Here are results (score, classification report, confusion matrix) for cross-validation set for 7 models tried:
model type: LogisticRegression; score: 0.899581589958159

              precision    recall  f1-score   support

       False       0.90      0.99      0.95       209
        True       0.80      0.27      0.40        30

    accuracy                           0.90       239
   macro avg       0.85      0.63      0.67       239
weighted avg       0.89      0.90      0.88       239

[[207   2]  
 [ 22   8]]

-------------------------------------

model type: RidgeClassifier; score: 0.8744769874476988

              precision    recall  f1-score   support

       False       0.87      1.00      0.93       209
        True       0.00      0.00      0.00        30

    accuracy                           0.87       239
   macro avg       0.44      0.50      0.47       239
weighted avg       0.76      0.87      0.82       239

[[209   0]  
 [ 30   0]]

-------------------------------------

model type: GaussianNB; score: 0.895397489539749

              precision    recall  f1-score   support

       False       0.99      0.89      0.94       209
        True       0.55      0.97      0.70        30

    accuracy                           0.90       239
   macro avg       0.77      0.93      0.82       239
weighted avg       0.94      0.90      0.91       239

[[185  24]  
 [  1  29]]

-------------------------------------

model type: MultinomialNB; score: 0.8744769874476988

              precision    recall  f1-score   support

       False       0.87      1.00      0.93       209
        True       0.00      0.00      0.00        30

    accuracy                           0.87       239
   macro avg       0.44      0.50      0.47       239
weighted avg       0.76      0.87      0.82       239

[[209   0]  
 [ 30   0]]

-------------------------------------

model type: BernoulliNB; score: 0.9330543933054394

              precision    recall  f1-score   support

       False       0.94      0.98      0.96       209
        True       0.82      0.60      0.69        30

    accuracy                           0.93       239
   macro avg       0.88      0.79      0.83       239
weighted avg       0.93      0.93      0.93       239

[[205   4]  
 [ 12  18]]

-------------------------------------

model type: CategoricalNB; score: 0.9497907949790795

              precision    recall  f1-score   support

       False       0.95      0.99      0.97       209
        True       0.91      0.67      0.77        30

    accuracy                           0.95       239
   macro avg       0.93      0.83      0.87       239
weighted avg       0.95      0.95      0.95       239

[[207   2]  
 [ 10  20]]

-------------------------------------

model type: KNeighborsClassifier; score: 0.899581589958159

              precision    recall  f1-score   support

       False       0.90      1.00      0.95       209
        True       1.00      0.20      0.33        30

    accuracy                           0.90       239
   macro avg       0.95      0.60      0.64       239
weighted avg       0.91      0.90      0.87       239

[[209   0]  
 [ 24   6]]

-------------------------------------

On the cross-validation set the two Naive Bayes models shows the best results by F1 score: GaussianNB, BernoulliNB, and CategoricalNB.

Since I'd rather have the model give false positive result (to have more candidates to then re-check by a human specialist) than false negative (missed potentially fitting candidates), the choice falls on the model with higher recall: GaussianNB.

The score for the best model so far on the test set is 0.839.

Classification report and confusion matrix are:
              precision    recall  f1-score   support

       False       0.99      0.83      0.90       268
        True       0.39      0.94      0.55        31

    accuracy                           0.84       299
   macro avg       0.69      0.88      0.72       299
weighted avg       0.93      0.84      0.87       299

[[222  46]  
 [  2  29]]

The precision is only 0.387, but most importantly the recall is 0.935.


Let's now try MinMaxScaler for features (Standard scaler uses negative values, and not all models support them).

Now the KNeighborsClassifier model not only ties the GaussianNB model in the recall metric, but also shows better performance in the precision metric.

model type: KNeighborsClassifier; score: 0.9790794979079498

              precision    recall  f1-score   support

       False       1.00      0.98      0.99       209
        True       0.88      0.97      0.92        30

    accuracy                           0.98       239
   macro avg       0.94      0.97      0.95       239
weighted avg       0.98      0.98      0.98       239

[[205   4]  
 [  1  29]]

Trying different K hyper-parameter values (from 1 to 20) showed that the highest recall is still for the default K=5.


The score for the best model so far on the test set is 0.973.

Classification report:
              precision    recall  f1-score   support

       False       0.99      0.98      0.98       268
        True       0.83      0.94      0.88        31

    accuracy                           0.97       299
   macro avg       0.91      0.96      0.93       299
weighted avg       0.98      0.97      0.97       299

Confusion matrix:  
[[262   6]  
 [  2  29]]

The precision is now 0.829, and the recall is still about the same: 0.935.



Test set (only first 30 examples):

|  | Age | BusinessTravel | Education | MaritalStatus | OverTime | Gender |
|---|---|---|---|---|---|---|
| 949 | 39 | 1 | 2 | 0 | 0 | 1 |
| 900 | 36 | 2 | 3 | 1 | 0 | 1 |
| 1338 | 30 | 1 | 3 | 0 | 0 | 1 |
| 983 | 34 | 1 | 4 | 0 | 0 | 0 |
| 958 | 34 | 1 | 3 | 2 | 0 | 1 |
| 561 | 52 | 1 | 4 | 1 | 0 | 1 |
| 481 | 34 | 1 | 2 | 1 | 1 | 1 |
| 303 | 31 | 1 | 3 | 1 | 0 | 1 |
| 342 | 31 | 1 | 4 | 0 | 1 | 0 |
| 244 | 45 | 1 | 3 | 1 | 0 | 1 |
| 1010 | 55 | 1 | 4 | 2 | 0 | 1 |
| 712 | 33 | 1 | 1 | 0 | 0 | 0 |
| 1137 | 22 | 0 | 2 | 1 | 1 | 0 |
| 1375 | 32 | 2 | 2 | 0 | 1 | 0 |
| 70 | 59 | 2 | 1 | 0 | 0 | 0 |
| 271 | 47 | 0 | 4 | 1 | 1 | 1 |
| 810 | 46 | 1 | 1 | 1 | 0 | 1 |
| 361 | 40 | 1 | 4 | 1 | 1 | 0 |
| 1213 | 23 | 1 | 3 | 2 | 1 | 1 |
| 51 | 28 | 1 | 4 | 0 | 1 | 1 |
| 423 | 30 | 0 | 4 | 1 | 0 | 0 |
| 777 | 21 | 1 | 3 | 0 | 0 | 0 |
| 184 | 53 | 1 | 2 | 2 | 0 | 0 |
| 744 | 37 | 1 | 2 | 1 | 0 | 0 |
| 381 | 30 | 1 | 1 | 1 | 0 | 1 |
| 759 | 45 | 1 | 4 | 0 | 0 | 1 |
| 1445 | 41 | 1 | 4 | 1 | 0 | 0 |
| 1463 | 31 | 0 | 3 | 0 | 0 | 1 |
| 1181 | 49 | 1 | 1 | 1 | 1 | 0 |
| 701 | 53 | 1 | 2 | 2 | 0 | 1 |

Some prediction examples for the test set: 
array([False, False, False, False, False, False, False, False,  True,
       False, False, False, False,  True, False, False, False, False,
        True,  True, False, False, False, False, False, False, False,
       False, False, False])

where the actual ground truth values are:
array([False, False, False, False, False, False, False, False,  True,
       False, False, False, False,  True, False, False, False, False,
        True,  True, False, False, False, False, False, False, False,
       False, False, False])
