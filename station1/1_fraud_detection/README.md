# Online Payments Fraud Detection

The model used in the project is sklearn.tree.DecisionTreeClassifier

The dataset has 6,354,407 samples, so it's OK to use 20% of its samples for testing.
However, positive samples take only 1.129% of the dataset, so I'm going to use precision, recall and F1 score to better evaluate the model.

Here are some observations regarding the features:
- isFlaggedFraud sounds like it's a label from another model, so dropping it.
- nameOrig, nameDest don't show anything useful, as they are unique identifiers, also dropping.
- isFraud is the label I'll use for learning.
- fraud is spread across many values of step, but for some step values there are little legit transactions.
- The type for fraudulent transaction is either CASH_OUT or TRANSFER (I'm converting this feature to one-hot set of features).
- Amount and oldbalanceDest are usually at lower bound, but they can also be legit transactions.
- Usually fraudulent transaction takes all the account money, so newbalanceOrig is zero often.

I'm leaving all the monetary features as they can help the model predict the fraud, and also step and type.

Here are the weights (importances) of the features after fitting the sklearn model:

| | |
| :--- | :--- |
| step | 0.064381 |
| amount | 0.148032 |
| oldbalanceOrg | 0.371469 |
| newbalanceOrig | 0.104659 |
| oldbalanceDest | 0.078031 |
| newbalanceDest | 0.209333 |
| type_CASH_IN | 0.000000 |
| type_CASH_OUT | 0.020422 |
| type_DEBIT | 0.000000 |
| type_PAYMENT | 0.000000 |
| type_TRANSFER | 0.003672 |

The accuracy of the model on the training set is 100%. For test data it's also close to 100% because positive cases are in minority.

Here are some better metrics to make sure the model is actually good at detecting positive cases:

| | |
| :--- | :--- |
| Precision | 0.898294 |
| Recall | 0.877778 |
| F1 Score | 0.887918 |

Which is a good result.
I've also tried class_weight="balanced" mode, but it gave slightly worse results by all of the metrics.
