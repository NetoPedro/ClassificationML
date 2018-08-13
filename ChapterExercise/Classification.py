from sklearn.datasets import fetch_mldata

# Loading the dataset
mnist = fetch_mldata('MNIST original')
print(mnist)

# Creating labels and data variables
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

# Analyzing the dataset

import matplotlib
import matplotlib.pyplot as plt

# some_digit = X[36000]
# some_digit_image = some_digit.reshape(28,28)
# plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
# plt.axis("off")
# plt.show()

# Splitting the dataset into a training set and a test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Shuffling the training set to avoid several consecutive similar images
import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Preparing data to try a binary classifier. To be 5 or not to be
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Preparing a Stochastic Gradient Descent Model
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42, max_iter=5)
sgd_clf.fit(X_train, y_train_5)

# Evaluating the performance
from sklearn.model_selection import cross_val_score

print("Cross_Validation Score -> ", cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# Getting predictions for every entry in the training set
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Getting a confusing matrix to analyze confused predictions
from sklearn.metrics import confusion_matrix

print("\nConfusion matrix \n", confusion_matrix(y_train_5, y_pred=y_train_pred), "\n\n")

# Calculating precision and recall
from sklearn.metrics import precision_score, recall_score

print("Precision = ", precision_score(y_train_5, y_pred=y_train_pred))
print("Recall = ", recall_score(y_train_5, y_pred=y_train_pred))

# Combing precision and recall into F1 Score
from sklearn.metrics import f1_score

print("F1 Score = ", f1_score(y_train_5, y_train_pred))

# Recall and Precision changing with the Threshold
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve

# Getting precision, recall and threshold values variations using scores from the decision function
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# Defining a method to plot the data
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# Ploting Precision vs Recall to choose the best tradeoff
def precision_vs_recall(precisions, recalls):
    plt.plot(recalls[:-1], precisions[:-1])
    plt.xlabel("Recall")
    plt.ylabel("precision")


precision_vs_recall(precisions, recalls)
plt.show()

# Calculating False Posite Rate and True Positive Rate for various thresholds
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# Ploting Roc curve
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positives Rate")
    plt.ylabel("True Positives Rate")


plot_roc_curve(fpr, tpr)
plt.show()

# Calculating Area under curve. Perfect Classifier has Area = 1
from sklearn.metrics import roc_auc_score

print("Area under curve = ", roc_auc_score(y_train_5, y_scores))

# Training a random forest for comparison porpuses
print("\n\nTraining a Random Forest\n\n")
from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(random_forest_classifier, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]

# Calculating metrics to plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores_forest)

plot_roc_curve(fpr, tpr)
plt.show()

# CAlculating Area Under the Curve for the ROC plot

print("Area under the curve = ", roc_auc_score(y_train_5, y_scores_forest))

# Precisions

random_forest_classifier.fit(X_train, y_train_5)
y_prediction_forest = cross_val_predict(random_forest_classifier, X_train, y_train_5, cv=3)
print("Precision = ", precision_score(y_train_5, y_prediction_forest))
print("Recall = ", recall_score(y_train_5, y_prediction_forest))
print("F1 Score = ", f1_score(y_train_5, y_prediction_forest))

# Training the Stochastic Gradient Descent instance in a multiclass dataset
print("\n\nTraining the Stochastic Gradient Descent instance in a multiclass dataset")
sgd_clf.fit(X_train, y_train)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

# Training the Random Forest instance in a multiclass dataset
print("\n\nTraining the Random Forest instance in a multiclass dataset")
random_forest_classifier.fit(X_train, y_train)
y_prediction_forest = cross_val_predict(random_forest_classifier, X_train, y_train_5, cv=3)

# Evaluating both classifiers
print("CrossValidation Score SGD = ", cross_val_score(sgd_clf,X_train,y_train, cv=3, scoring="accuracy"))
print("CrossValidation Score Random Forest = ", cross_val_score(random_forest_classifier,X_train,y_train, cv=3, scoring="accuracy"))

# Applying a scaling technique to evaluate effects on the performance
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train.astype(np.float64))

print("CrossValidation Score Scaled SGD = ", cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
print("CrossValidation Score Scaled Random Forest = ", cross_val_score(random_forest_classifier, X_train_scaled, y_train, cv=3, scoring="accuracy"))

# Generating a confusion matrix for the stochastic gradient descent instance
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled,y_train,cv=3)
conf_matrix = confusion_matrix(y_train,y_train_pred)
print(conf_matrix)

# Generating an image representation of the matrix
plt.matshow(conf_matrix,cmap=plt.cm.gray)
plt.show()

# Focusing on errors. Therefore removing correct predictions and creating relative errors values

row_sums = conf_matrix.sum(axis=1,keepdims=True)
norm_conf_matrix = conf_matrix/row_sums

np.fill_diagonal(norm_conf_matrix,0)
plt.matshow(norm_conf_matrix,cmap=plt.cm.gray)
plt.show()

# Multilabel Classifications with K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train>=7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large,y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)

