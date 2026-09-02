# Pre-interview assessment

The project is aimed to predict the HR decision regarding offering a new candidate a company position based on previously accumulated statistics.

The Python notebook can be viewed right in this Github repository. I've used anaconda.com/app to create it, but probably it will work in other flavors of Python notebooks too.

In this project I try several models from scikit-learn: 4 Naive Bayes models, 2 Logistic Regression models, and one K-nearest-neighbors.

All is done using the standard numpy+matplotlib+pandas stack.

Based on plots of different features, this company prefers candidates who are not married, and are OK with business trips and overtime (perhaps, business trips are an essential part of the job).

Education and age also are an important factor for the decision.

Since OverTime, Gender, BusinessTravel, and MaritalStatus are given as strings, I'm using OrdinalEncoder to convert them to numbers.

Here are results (score, classification report, confusion matrix) for cross-validation set for 7 models tried:

### LogisticRegression
**Score:** 0.8996

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 0.90 | 0.99 | 0.95 | 209 |
| True | 0.80 | 0.27 | 0.40 | 30 |
| **Accuracy** | | | **0.90** | 239 |
| Macro avg | 0.85 | 0.63 | 0.67 | 239 |
| Weighted avg | 0.89 | 0.90 | 0.88 | 239 |

<br>

**Confusion matrix**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 207 | 2 |
| **Actual True** | 22 | 8 |

-------------------------------------

### RidgeClassifier
**Score:** 0.8745

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 0.87 | 1.00 | 0.93 | 209 |
| True | 0.00 | 0.00 | 0.00 | 30 |
| **Accuracy** | | | **0.87** | 239 |
| Macro avg | 0.44 | 0.50 | 0.47 | 239 |
| Weighted avg | 0.76 | 0.87 | 0.82 | 239 |

<br>

**Confusion matrix**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 209 | 0 |
| **Actual True** | 30 | 0 |

-------------------------------------

### GaussianNB
**Score:** 0.8954

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 0.99 | 0.89 | 0.94 | 209 |
| True | 0.55 | 0.97 | 0.70 | 30 |
| **Accuracy** | | | **0.90** | 239 |
| Macro avg | 0.77 | 0.93 | 0.82 | 239 |
| Weighted avg | 0.94 | 0.90 | 0.91 | 239 |

<br>

**Confusion matrix**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 185 | 24 |
| **Actual True** | 1 | 29 |

-------------------------------------

### MultinomialNB
**Score:** 0.8745

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 0.87 | 1.00 | 0.93 | 209 |
| True | 0.00 | 0.00 | 0.00 | 30 |
| **Accuracy** | | | **0.87** | 239 |
| Macro avg | 0.44 | 0.50 | 0.47 | 239 |
| Weighted avg | 0.76 | 0.87 | 0.82 | 239 |

<br>

**Confusion matrix**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 209 | 0 |
| **Actual True** | 30 | 0 |

-------------------------------------

### BernoulliNB
**Score:** 0.9331

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 0.94 | 0.98 | 0.96 | 209 |
| True | 0.82 | 0.60 | 0.69 | 30 |
| **Accuracy** | | | **0.93** | 239 |
| Macro avg | 0.88 | 0.79 | 0.83 | 239 |
| Weighted avg | 0.93 | 0.93 | 0.93 | 239 |

<br>

**Confusion matrix**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 205 | 4 |
| **Actual True** | 12 | 18 |

-------------------------------------

### CategoricalNB
**Score:** 0.9498

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 0.95 | 0.99 | 0.97 | 209 |
| True | 0.91 | 0.67 | 0.77 | 30 |
| **Accuracy** | | | **0.95** | 239 |
| Macro avg | 0.93 | 0.83 | 0.87 | 239 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 239 |

<br>

**Confusion matrix**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 207 | 2 |
| **Actual True** | 10 | 20 |

-------------------------------------

### KNeighborsClassifier
**Score:** 0.8996

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 0.90 | 1.00 | 0.95 | 209 |
| True | 1.00 | 0.20 | 0.33 | 30 |
| **Accuracy** | | | **0.90** | 239 |
| Macro avg | 0.95 | 0.60 | 0.64 | 239 |
| Weighted avg | 0.91 | 0.90 | 0.87 | 239 |

<br>

**Confusion matrix**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 209 | 0 |
| **Actual True** | 24 | 6 |

-------------------------------------

On the cross-validation set the two Naive Bayes models shows the best results by F1 score: GaussianNB, BernoulliNB, and CategoricalNB.

Since I'd rather have the model give false positive result (to have more candidates to then re-check by a human specialist) than false negative (missed potentially fitting candidates), the choice falls on the model with higher recall: GaussianNB.

The score for the best model so far on the test set is 0.839.

Classification report and confusion matrix are:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 0.99 | 0.83 | 0.90 | 268 |
| True | 0.39 | 0.94 | 0.55 | 31 |
| **Accuracy** | | | **0.84** | 299 |
| Macro avg | 0.69 | 0.88 | 0.72 | 299 |
| Weighted avg | 0.93 | 0.84 | 0.87 | 299 |

<br>

**Confusion matrix**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 222 | 46 |
| **Actual True** | 2 | 29 |

The precision is only 0.387, but most importantly the recall is 0.935.


Let's now try MinMaxScaler for features (Standard scaler uses negative values, and not all models support them).

Now the KNeighborsClassifier model not only ties the GaussianNB model in the recall metric, but also shows better performance in the precision metric.

### KNeighborsClassifier (MinMaxScaler)
**Score:** 0.9791

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 1.00 | 0.98 | 0.99 | 209 |
| True | 0.88 | 0.97 | 0.92 | 30 |
| **Accuracy** | | | **0.98** | 239 |
| Macro avg | 0.94 | 0.97 | 0.95 | 239 |
| Weighted avg | 0.98 | 0.98 | 0.98 | 239 |

<br>

**Confusion matrix**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 205 | 4 |
| **Actual True** | 1 | 29 |

Trying different K hyper-parameter values (from 1 to 20) showed that the highest recall is still for the default K=5.


The score for KNeighborsClassifier on the test set is 0.973.

**Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| False | 0.99 | 0.98 | 0.98 | 268 |
| True | 0.83 | 0.94 | 0.88 | 31 |
| **Accuracy** | | | **0.97** | 299 |
| Macro avg | 0.91 | 0.96 | 0.93 | 299 |
| Weighted avg | 0.98 | 0.97 | 0.97 | 299 |

<br>

**Confusion matrix:**

|  | Pred False | Pred True |
|---|---|---|
| **Actual False** | 262 | 6 |
| **Actual True** | 2 | 29 |

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

```
array([False, False, False, False, False, False, False, False,  True,
       False, False, False, False,  True, False, False, False, False,
        True,  True, False, False, False, False, False, False, False,
       False, False, False])
```

where the actual ground truth values are:

```
array([False, False, False, False, False, False, False, False,  True,
       False, False, False, False,  True, False, False, False, False,
        True,  True, False, False, False, False, False, False, False,
       False, False, False])
```
