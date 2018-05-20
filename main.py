import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('pulsar_stars.csv', header=None)
df.columns = df.iloc[0, :].values
print(df.columns)
feat_labels = df.columns[:8]

X, y = df.iloc[1:, :8].values, df.iloc[1:, 8].values
X, y = X.astype('float64'), y.astype('float64')


# %30 test %70 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)


"""
# Find best attributes with Random Forest
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))


X = df.loc[1:, (' Excess kurtosis of the integrated profile', ' Skewness of the integrated profile')].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
"""




pipe_lr = make_pipeline(StandardScaler(),
                      PCA(n_components=2),
                      LogisticRegression(random_state=1))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Logistic Regression: %', accuracy_score(y_test, y_pred))

pipe_svm = make_pipeline(StandardScaler(),
                         PCA(n_components=2),
                         SVC(random_state=1))
pipe_svm.fit(X_train, y_train)
y_pred = pipe_svm.predict(X_test)
print('SVM : %', accuracy_score(y_test, y_pred))

pipe_forest = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            RandomForestClassifier(n_estimators=100, random_state=1))
pipe_forest.fit(X_train, y_train)
y_pred = pipe_forest.predict(X_test)
print('Random Forest: %', accuracy_score(y_test, y_pred))

pipe_knn = make_pipeline(StandardScaler(),
                         PCA(n_components=2),
                         KNeighborsClassifier(n_neighbors=5))
pipe_knn.fit(X_train, y_train)
y_pred = pipe_knn.predict(X_test)
print('KNN : %', accuracy_score(y_test, y_pred))

kfold = StratifiedKFold(n_splits= 10,
                        random_state=1).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X[train], y[train])
    score = pipe_lr.score(X[test], y[test])
    scores.append(score)
    print('Fold: %2d, Acc: %.3f' % (k + 1, score))

print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))



