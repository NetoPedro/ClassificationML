import pandas
import numpy as np

train_set = pandas.read_csv("./dataset/train.csv")
test_set = pandas.read_csv("./dataset/test.csv")
print(train_set)
train_set = train_set.drop('id',axis=1)
print(train_set.describe())

train_set['type'], categories = train_set['type'].factorize()

import matplotlib.pyplot as plt
print(train_set.info())
'''
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
cax = ax.matshow(train_set.corr())
fig.colorbar(cax)

ax.set_xticklabels(train_set.columns)
ax.set_yticklabels(train_set.columns)

plt.show()'''

X_train = train_set.drop('type',axis=1)
y_train = train_set.get('type')
X_train= X_train.append(test_set)
#print(X_train.info())

from sklearn.base import BaseEstimator, TransformerMixin

class CreateExtraFeatures(BaseEstimator,TransformerMixin):
    def __init__(self):pass

    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X['hair_soul'] = X['hair_length'] * X['has_soul']
        X['flesh_soul'] = X['rotting_flesh'] * X['has_soul']
        return np.c_[X]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
num_attributes = ["bone_length","rotting_flesh","hair_length","has_soul"]
cat_attributes = ["color"]

pipeline_num = Pipeline([
    ("selector",DataFrameSelector(num_attributes)),
    ("extra_feat",CreateExtraFeatures())
])

pipeline_cat = Pipeline([
    ("selector", DataFrameSelector(cat_attributes)),
    ("categorical_encoder", OneHotEncoder(sparse=False))
])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion([
    ("pip,num",pipeline_num),
    ("pip_cat",pipeline_cat)
])
X_train= full_pipeline.fit_transform(X_train)

X_test = X_train[371:]
X_train = X_train[:371]
from sklearn.neural_network import MLPClassifier

nn_clf = MLPClassifier(max_iter=3000)

from sklearn.model_selection import GridSearchCV

grid_params = [{"hidden_layer_sizes":range(3,20), "activation":['identity', 'logistic', 'tanh', 'relu'], "solver":["lbfgs","sgd","adam"],"learning_rate":["adaptive"]}]
grid_search = GridSearchCV(nn_clf,param_grid=grid_params,cv=3,verbose=3, n_jobs=-1)

grid_search.fit(X_train,y_train)

print(grid_search.best_estimator_)
print(grid_search.best_score_)

#X_test = full_pipeline.fit_transform(test_set[num_attributes],test_set[cat_attributes].values)



y_pred = grid_search.predict(X_test)

submissions = pandas.DataFrame(y_pred, index=test_set.id,columns=["type"])
submissions["type"] = categories[submissions["type"]]
submissions.to_csv('submission.csv', index=True)
