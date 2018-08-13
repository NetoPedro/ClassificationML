import pandas
import numpy as np

# Importing the training set
train_set =  pandas.read_csv("./dataset/train.csv")
#print(train_set)
# Splitting X and y
X_train = train_set.drop(['Survived','PassengerId','Name','Ticket'], axis=1)
#print(X_train)

y_train = train_set.drop(['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
#print(y_train)

#Loading Test set

test_set =  pandas.read_csv("./dataset/test.csv")
#print(train_set)
# Splitting X and y
X_test = test_set.drop(['PassengerId','Name','Ticket'], axis=1)

X_train['Age'] = X_train['Age']/15
X_test['Age'] = X_test['Age']/15

cabin = X_train['Cabin']
cabin.fillna('0')
cabin[cabin != '0'] = '1'
X_train['Cabin'] = cabin.astype(int)

cabin = X_test['Cabin']
cabin.fillna('0')
cabin[cabin != '0'] = '1'
X_test['Cabin'] = cabin.astype(int)

alone =  X_train['SibSp'] + X_train['Parch'] + 1
alone[alone > 1] = 0
X_train['is_alone'] = alone

alone =  X_test['SibSp'] + X_test['Parch'] + 1
alone[alone > 1] = 0
X_test['is_alone'] = alone


# Analyzing data
#print(X_train.info())

#print(train_set.corr()['Survived'])



import  matplotlib.pyplot as plt
X_train.hist(bins=50,figsize=(10,8))
#plt.show()

from sklearn.base import  BaseEstimator, TransformerMixin

class DataPreProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        # Dealing with non numerical attributes
        # Converting Sex in a numerical attribute
        X_train_non_num = X.drop(['Cabin','Age', 'SibSp', 'Parch', 'Fare'], axis=1)
        X_train_encoded, X_train_categories = X_train_non_num['Sex'].factorize()
        # print(X_train_encoded)



        X['Sex'] = X_train_encoded
        #print(X)

        # Converting Embarked into numerical attributes
        X_train_encoded, X_train_categories = X_train_non_num['Embarked'].factorize()
        # print(X_train_encoded)
        X_train_encoded = X_train_encoded.astype(float)

        # Missing values will be treated with the use of an imputer
        X_train_encoded[X_train_encoded == -1] = np.NAN
        # print(X_train_encoded)

        from sklearn.preprocessing import Imputer
        imputer = Imputer(strategy="mean")
        X_train_encoded = imputer.fit_transform(X_train_encoded.reshape(-1, 1))

        # Categories will be divided in columns
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder()
        X_train_1_hot = encoder.fit_transform(X_train_encoded)
        # print(X_train_1_hot.toarray())
        X = X.drop('Embarked', axis=1)
        X['S'] = X_train_1_hot[:, 0].toarray().astype(int)
        X['C'] = X_train_1_hot[:, 1].toarray().astype(int)
        X['Q'] = X_train_1_hot[:, 2].toarray().astype(int)


        #Encoding Pclass
        X_train_encoded, X_train_categories = X_train_non_num['Pclass'].factorize()

        encoder = OneHotEncoder()
        X_train_1_hot = encoder.fit_transform(X_train_encoded.reshape(-1,1))
        # print(X_train_1_hot.toarray())
        X = X.drop('Pclass', axis=1)
        X['1'] = X_train_1_hot[:, 0].toarray().astype(int)
        X['2'] = X_train_1_hot[:, 1].toarray().astype(int)
        X['3'] = X_train_1_hot[:, 2].toarray().astype(int)

        # Imputer will now run for every feature

        aux = imputer.fit_transform(X)
        X = pandas.DataFrame(aux, columns=X.columns)

        # print(X_train)

        # Scaling attributes

        '''from sklearn.preprocessing import StandardScaler

        std_scaler = StandardScaler()

        aux = std_scaler.fit_transform(X)
        X = pandas.DataFrame(aux, columns=X.columns)
        '''
        return X




from sklearn.pipeline import  Pipeline

pipeline = Pipeline([
    ('general',DataPreProcessing())
])

aux = pipeline.fit_transform(X_train)
X_train = pandas.DataFrame(aux)
#print(X_train)

aux = pipeline.fit_transform(X_test)
X_test = pandas.DataFrame(aux)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Training a Random Forest
print(X_train)
random_forest = RandomForestClassifier(random_state=None)
param_grid_forest = [{'max_depth': range(5,15),"n_estimators":range(5,50,5),"max_features":['log2', 'sqrt', 'auto'],"min_samples_split":range(2,4)}]
grid_forest = GridSearchCV(random_forest,param_grid=param_grid_forest,scoring=None,cv=3,verbose=3)
grid_forest.fit(X_train,y_train.values.ravel())
print(grid_forest.best_estimator_)
print(grid_forest.best_score_)


y_pred = grid_forest.predict(X_test)

submissions = pandas.DataFrame(y_pred, index=test_set.PassengerId,columns=["Survived"])
submissions.to_csv('submission.csv', index=True)
'''

from sklearn.ensemble import BaggingClassifier
param_grid_forest = [{"n_estimators":range(20,50,5),"max_features":range(10,11),'max_samples':range(19,30),"oob_score":[True,False]}]
bagg = BaggingClassifier()
grid_forest = GridSearchCV(bagg,param_grid=param_grid_forest,scoring=None,cv=3,verbose=3)
grid_forest.fit(X_train,y_train.values.ravel())
print(grid_forest.best_estimator_)
print(grid_forest.best_score_)
y_pred = grid_forest.predict(X_test)

submissions = pandas.DataFrame(y_pred, index=test_set.PassengerId,columns=["Survived"])
submissions.to_csv('submission.csv', index=True)

svm = SVC()


grid_params = [{"kernel":["linear","poly","rbf","sigmoid"],"degree":[1,2,3,4,6],"coef0":[0.0,1,2],"gamma":np.arange(1e-4,1e-2,0.0001),"probability":[True,False]}]
grid_search = GridSearchCV(svm,grid_params,n_jobs=-1,cv=3,verbose=3)
grid_search.fit(X_train,y_train.values.ravel())
print(grid_search.best_estimator_)
print(grid_search.best_score_)
'''