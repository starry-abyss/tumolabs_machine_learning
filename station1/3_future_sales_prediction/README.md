# Future Sales Prediction

We have datapoints for the Sales label with respect to the amount of advertisement budget per platform for the same time period.

The platforms (from ML perspective they are features) for which we data are the following:
* TV
* Radio
* Newspaper

We can plot scatters of each feature vs the Sales label and see that they looks like a straight line + some deviation, so we choose Linear Regression.

Let's use a standard split of 20% for test data. We are going to import the Linear Regression model from the scikit-learn package.

After fitting we can plot the scatters again with the model line on top. There is more correlation of Sales with TV and Radio a campaigns and little to none - with Newspaper ads.

The resulting precision is around 91% for the training set, and around 89% for the test set, which also looks realistic and good.

I think it makes sense to firstly cut the budget of Newspaper ads because its influence on the resulting sales doesn't increase very much by increasing the spendings. Instead the focus should be more on the balance between TV and Radio ads.
