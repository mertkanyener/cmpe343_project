import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Find best two attributes with Random Forest Classifier
# and return X matrix with only best two attributes

def read_data(filename):

    df = pd.read_csv('pulsar_stars.csv', header=None)
    df.columns = df.iloc[0, :].values
    feat_labels = df.columns[:8]

    X, y = df.iloc[1:, :8].values, df.iloc[1:, 8].values
    X, y = X.astype('float64'), y.astype('float64')

    return X, y, feat_labels, df

def find_best_attr(X, y, df, feat_labels):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

    # Find best attributes with Random Forest
    forest = RandomForestClassifier(n_estimators=500, random_state=1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    c = 1
    print("Best Attributes: ")
    for f in range(X_train.shape[1]):
        if c == 1:
            col1 = feat_labels[indices[f]]
        elif c == 2:
            col2 = feat_labels[indices[f]]
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))
        c += 1
    X_best = df.loc[1:, (col1, col2)].values
    X_best = X_best.astype('float64') # to avoid conversion warnings

    return X_best


def make_pipelines():

    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(random_state=1))

    pipe_svm = make_pipeline(StandardScaler(),
                             PCA(n_components=2),
                             SVC(random_state=1))

    pipe_forest = make_pipeline(StandardScaler(),
                                PCA(n_components=2),
                                RandomForestClassifier(n_estimators=100, random_state=1))

    pipe_knn = make_pipeline(StandardScaler(),
                             PCA(n_components=2),
                             KNeighborsClassifier(n_neighbors=5))

    pipelines = pipe_lr, pipe_svm, pipe_forest, pipe_knn

    return pipelines



# Run pipelines for 4 classification algorithms, data split by %80 train, %20 test

def run_pipes(X, y, pipelines):


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
    pipe_lr, pipe_svm, pipe_forest, pipe_knn = pipelines

    print("Logistic Regression")

    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Confusion Matrix: ")
    print(confmat)
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

    # SVM

    print("----------------------------")
    print("SVM")


    pipe_svm.fit(X_train, y_train)
    y_pred = pipe_svm.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Confusion Matrix: ")
    print(confmat)
    print('Accuracy : %.3f' % accuracy_score(y_test, y_pred))

    # Random Forest Classifier

    print("----------------------------")
    print("Random Forest Classifier")


    pipe_forest.fit(X_train, y_train)
    y_pred = pipe_forest.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Confusion Matrix: ")
    print(confmat)
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

    # KNN

    print("----------------------------")
    print("KNN")


    pipe_knn.fit(X_train, y_train)
    y_pred = pipe_knn.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Confusion Matrix: ")
    print(confmat)
    print('Accuracy : %.3f' % accuracy_score(y_test, y_pred))


# Run algorithms and calculate their accuracies with kfold cross validation

def run_pipes_kfold(X, y, pipelines):
    # Logistic Regression (kfold)

    pipe_lr, pipe_svm, pipe_forest, pipe_knn = pipelines
    kfold = StratifiedKFold(n_splits=10,
                            random_state=1).split(X, y)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X[train], y[train])
        score = pipe_lr.score(X[test], y[test])
        scores.append(score)
        # print('Fold: %2d, Acc: %.3f' % (k + 1, score))

    print('\nLogistic Regression CV accuracy: %.3f' % np.mean(scores))

    # SVM (kfold)

    kfold = StratifiedKFold(n_splits=10,
                            random_state=1).split(X, y)

    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_svm.fit(X[train], y[train])
        score = pipe_svm.score(X[test], y[test])
        scores.append(score)
        # print('Fold: %2d, Acc: %.3f' % (k + 1, score))

    print('\nSVM CV accuracy: %.3f' % np.mean(scores))

    # Random Forest Classifier

    kfold = StratifiedKFold(n_splits=10,
                            random_state=1).split(X, y)

    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_forest.fit(X[train], y[train])
        score = pipe_forest.score(X[test], y[test])
        scores.append(score)
        # print('Fold: %2d, Acc: %.3f' % (k + 1, score))

    print('\nRandom Forest CV accuracy: %.3f' % np.mean(scores))

    # KNN (kfold)

    kfold = StratifiedKFold(n_splits=10,
                            random_state=1).split(X, y)

    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_knn.fit(X[train], y[train])
        score = pipe_knn.score(X[test], y[test])
        scores.append(score)
        # print('Fold: %2d, Acc: %.3f' % (k + 1, score))

    print('\nKNN CV accuracy: %.3f' % np.mean(scores))


X, y, feat_labels, df = read_data('pulsar_stars.csv')
pipelines = make_pipelines()
# Run algorithms with all attributes
print("Results with all attributes: ")
run_pipes(X, y, pipelines)
run_pipes_kfold(X, y, pipelines)

# Run all algorithms with best two attributes

print("--------------------------")
print("Results with two best attributes: ")
X_best = find_best_attr(X, y, df, feat_labels)
run_pipes(X_best, y, pipelines)
run_pipes_kfold(X_best, y,  pipelines)





