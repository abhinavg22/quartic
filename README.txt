Conceptual Approach:

The aim of the problem was to predict the target column of the test dataset. On evaluating the training dataset, there were 56 variables 
which were divided into 3 categories,i.e., numerical, derived and categorical, all having integer or float values. So, to find the 
relation between all the variables and target column, I used correlation matrix funtion, but the difference was not very stark. Hence, I
decided to use PCA to reduce dimensions. 

Training dataset had large amount of missing values for cat6 column and cat8 column. So, to fill in the missing values, I used Imputer 
function in the pipeline.

To predict the target column, I used various models such as Logistic Regression, Random forest and Gradient Boost.
Gradient boost performed the best amongst the models as I was able to tune the hyperparameter.


Trade Off:
As there was not much information available about the variables, there was no way to decide which features/columns were significant and 
including all the features meant large running time for algorithms.Therefore, I went for dimensionality reduction using PCA which led 
to my algorithms' accuracy to reduce by a small margin.
Hence, I had to trade-off between running time of algorithm and algorithm's accuracy.


Model Performance:
The mean score of cross validation for various algorithms used are as follows:
Logistic Regression: 0.96356375
Random Forest: 0.963578859
Gradient Boost: 0.95935906


Complexity & Bottlenecks:
The algorithms were taking too much time to run and generate cross validation scores as the dataset had many variables.
Also, unavailability of insights on the variables/columns didn't allow me to gauge their significance with respect to the target column.


Improvements:
Given more time, I would like to get more information on the various columns so as to find their relation with the target column.
I would have built a neural network structure on the dataset which could have generated better results but considering the number of 
features involved and using my personal machine for the same, it would have taken a very large amount of time.