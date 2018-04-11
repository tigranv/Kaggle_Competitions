from pandas import read_table , get_dummies, concat , read_csv
import numpy as np
import matplotlib.pyplot as plt
import csv

# =====================================================================
test = read_csv("../KaggleCompetitions/Titanic/reducedData/test_reduced.csv", encoding='latin-1', sep=',', skipinitialspace=True, index_col=None, header=0,)
train = read_csv("../KaggleCompetitions/Titanic/reducedData/train_reduced.csv", encoding='latin-1', sep=',', skipinitialspace=True, index_col=None, header=0)
all = read_csv("../KaggleCompetitions/Titanic/OriginalData/train.csv", encoding='latin-1', sep=',', skipinitialspace=True, index_col=None, header=0,)

targets = all.Survived

test.head()
train.head()
targets.head()

X_train = np.array(train, dtype=np.float)
X_test = np.array(test, dtype=np.float)
y_train = np.array(targets, dtype=np.float)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

# Some classifiers to test
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Test the DecisionTreeClassifier classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
score = f1_score(y_test, classifier.predict(X_test))
print("DecisionTreeClassifier - {}".format(score))

# Test the Ada boost classifier
classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
classifier.fit(X_train, y_train)
Submission_pred = classifier.predict(X_Sub)
with open('../KaggleCompetitions/Titanic/AdaBoostsubmission.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['PassengerId'] + ['Survived'])
    for i in range(len(Submission_pred)):
        spamwriter.writerow([892 + i] + [int(Submission_pred[i])])



# Test Random Forest

random_forest = RandomForestClassifier(n_estimators=100, max_features='sqrt')
random_forest.fit(X_train, y_train)

Submission_pred = random_forest.predict(X_Sub)
with open('../KaggleCompetitions/Titanic/gender_submissionInAlldata.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['PassengerId'] + ['Survived'])
    for i in range(len(Submission_pred)):
        spamwriter.writerow([892 + i] + [int(Submission_pred[i])])






