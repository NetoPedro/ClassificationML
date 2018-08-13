
import multiprocessing

multiprocessing.set_start_method('forkserver')

from sklearn.datasets import fetch_mldata

# Loading the dataset
mnist = fetch_mldata('MNIST original')
print(mnist)

# Creating labels and data variables
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Randomizing training set order
import numpy as np

shuffled_indexes = np.random.permutation(60000)
X_train, y_train = X_train[shuffled_indexes], y_train[shuffled_indexes]


# Simple training of a K Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=-1)
knn_clf.fit(X_train, y_train)

# Calculation of accuracy
from sklearn.metrics import accuracy_score
#knn_pred = knn_clf.predict(X_test)
#accuracy_score = accuracy_score(y_test,knn_pred)
#print(accuracy_score)

from sklearn.model_selection import GridSearchCV

grid_params = [{"weights":["distance","uniform"],"n_neighbors":[2,4,8]}]

grid_search = GridSearchCV(knn_clf,grid_params,n_jobs=-1,cv=3,verbose=3)

grid_search.fit(X_train ,y_train)

print(grid_search.best_params_)

print(grid_search.best_score_)

grid_pred = grid_search.predict(X_test)

print("Accuracy = ", accuracy_score(y_test,grid_pred))