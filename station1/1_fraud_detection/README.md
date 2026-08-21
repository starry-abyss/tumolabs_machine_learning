1. Online Payments Fraud Detection

The model used in the project is sklearn.tree.DecisionTreeClassifier

The dataset has 6,354,407 samples, so it's OK to use 20% of its samples for testing.
However, positive samples take only 0.129% of the dataset, so I'm using class_weight="balanced" mode to give more weight to positive samples.

Here are some observations regarding the features:

isFlaggedFraud sounds like it's a label from another model, so dropping it.
nameOrig, nameDest don't show anything useful, as they are unique identifiers, also dropping.
isFraud is the label I'll use for learning.

fraud is spread across many values of step, but for some step values there are little legit transactions.

The type for fraudulent transaction is either CASH_OUT or TRANSFER (I'm converting strings to numeric classes).

Amount and oldbalanceDest are usually at lower bound, but they can also be legit transactions.

Usually fraudulent transaction takes all the account money, so newbalanceOrig is zero often.

I'm leaving all the monetary features as they can help the model predict the fraud, and also step and type.

Here are the weights (importances) of the features after fitting the sklearn model:
step               0.038501
amount             0.156324
oldbalanceOrg      0.356041
newbalanceOrig     0.371845
oldbalanceDest     0.002576
newbalanceDest     0.051613
transactionType    0.023100

The accuracy of the model on the training set if 100%. From what I know, it's common for this type of model, and I don't possess deep knowledge of this model as of learning station 1 in order to tune it to not overfit the dataset.

For test data it's also close to 100% because positive cases are in minority, but when giving only positive input, the accuracy is 85.7%, which looks good to me.
