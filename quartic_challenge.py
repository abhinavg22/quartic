#Path directory 
import os
Data_Path = os.path.join('F:\Hiring Challenge', 'ds_data')

#Function to extract from all the paths 
import pandas as pd
def load_data(filename, path = Data_Path):
    filepath = os.path.join(path, filename)
    return pd.read_csv(filepath) 

train = load_data('data_train.csv')
testf = load_data('data_test.csv')
train.describe()

corr_matrix = train.corr()
corr_matrix['target'].sort_values(ascending=False)
train.info()

#Transformer to handle Pandas Dataframas
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
	
	
y_train = train['target']
train = train.drop('target', axis=1)
train.head()

y_train.value_counts()

#Pipeline to fill in the missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler

imputer = Imputer(strategy='median')

pipeline = Pipeline([
    ('selector', DataFrameSelector(['num1','num2','num3','num4','num5','num6','num7','num8','num9','num10','num11','num12',
                                    'num13','num14','num15','num16','num17','num18','num19','num20','num21','num22','num23',
                                   'der1','der2','der3','der4','der5','der6','der7','der8','der9','der10','der11','der12',
                                   'der13','der14','der15','der16','der17','der18','der19','cat1','cat2','cat3','cat4',
                                   'cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14'])),
    ('imputer', Imputer(strategy='median')),
    ('std_scaler',StandardScaler()),
 ])

X_train = pipeline.fit_transform(train)

#Using PCA to reduce the variables
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

X_test = pipeline.transform(testf)
X_test_reduced = pca.fit_transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(random_state=42)
log_clf.fit(X_reduced,y_train)

#Mean Score of Logistic Regression
from sklearn.model_selection import cross_val_score
log_scores = cross_val_score(log_clf,X_reduced,y_train, cv=5)
log_scores.mean()

#Random Forest
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_reduced, y_train)

#Mean Score of Random Forest
from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest_clf, X_reduced, y_train, cv=5)
forest_scores.mean()

#Gradient Boost
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(n_estimators=600, learning_rate=0.5, max_depth=3, random_state=0)
gb_clf.fit(X_reduced, y_train)

#Mean Score of Gradient Boost
from sklearn.model_selection import cross_val_score
gb_scores = cross_val_score(gb_clf, X_reduced, y_train, cv = 5)
gb_scores.mean()

#Using Gradient Boost to predict the test target variable
y_test_gb = gb_clf.predict(X_test_reduced)

pred = {'target' : y_test_gb}
target = pd.DataFrame(pred)
print(target)

#Test data with predicted target variable
final = pd.concat([testf, target], axis= 1, join = 'inner')

print(final.head())

result = final[['id', 'target']]
print(result)

#Output file
result.to_csv('F:\Challenge Outputs\Quartic.ai Hiring Challenge.csv')

result['target'].value_counts()