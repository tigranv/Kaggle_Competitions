from pandas import read_table , get_dummies, concat
import numpy as np
import matplotlib.pyplot as plt
import csv

def download_data():

    frame = read_table("../KaggleCompetitions/Titanic/train.csv", encoding='latin-1', sep=',', skipinitialspace=True, index_col=None, header=0)

    return frame

def download_submission_data():
    

    frame = read_table("../KaggleCompetitions/Titanic/test.csv", encoding='latin-1', sep=',', skipinitialspace=True, index_col=None, header=0,)

    return frame

# =====================================================================


def get_features_and_labels(frame, frame_sub):

    frame_lable = frame.iloc[:, -1]
    frame_train = frame.iloc[:, :-1]
    frame_features = frame_train.append(frame_sub)[[ "Pclass", "Sex", "Age", "Fare", "Embarked"] ]
    #frame_features = frame_features.fillna(0)

    y = np.array(frame_lable, dtype=np.int)

    frame_features = get_dummies(frame_features, columns=["Embarked"])
    X_All = np.array(frame_features, dtype=np.float)
    X = X_All[:891, :]
    X_Sub = X_All[891:,:]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)
 
    # If values are missing we could impute them from the training data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy='mean')
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    X_Sub = imputer.transform(X_Sub)
    # Normalize the attribute values to mean=0 and variance=1
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    # To scale to a specified range
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # scaling to both training and test sets.
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_Sub = scaler.transform(X_Sub)

    return X_train, X_test, y_train, y_test, X_Sub


# =====================================================================

def evaluate_classifier(X_train, X_test, y_train, y_test, X_Sub):

    # Some classifiers to test
    from sklearn.svm import LinearSVC, NuSVC, SVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    from sklearn.metrics import precision_recall_curve, f1_score
    
    
    # Test the linear support vector classifier
    classifier = LinearSVC(C=1)
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    print("LinearSVC - {}".format(score))
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    yield 'Linear SVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the KNeighborsClassifier classifier
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    print("KNeighborsClassifier - {}".format(score))

    # Test the DecisionTreeClassifier classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    print("DecisionTreeClassifier - {}".format(score))

    # Test the Nu support vector classifier
    classifier = NuSVC(kernel='rbf', nu=0.5, gamma=1e-3)
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    print("NuSVC - {}".format(score))
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    yield 'NuSVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Ada boost classifier
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    print("AdaBoostClassifier - {}".format(score))
    Submission_pred = classifier.predict(X_Sub)
    with open('../KaggleCompetitions/Titanic/gender_submission1.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['PassengerId'] + ['Survived'])
        for i in range(len(Submission_pred)):
            spamwriter.writerow([892 + i] + [int(Submission_pred[i])])
            #print("{} - {}".format(892+i, int(Submission_pred[i])))
        
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    yield 'Ada Boost (F1 score={:.3f})'.format(score), precision, recall


    random_forest = RandomForestClassifier(n_estimators=100, max_features='sqrt')
    random_forest.fit(X_train, y_train)

    Submission_pred = random_forest.predict(X_Sub)
    with open('../KaggleCompetitions/Titanic/gender_submissionInAlldata.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['PassengerId'] + ['Survived'])
        for i in range(len(Submission_pred)):
            spamwriter.writerow([892 + i] + [int(Submission_pred[i])])
# =====================================================================


def plot(results):

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data ')

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left')

    plt.tight_layout()

    plt.show()
    plt.close()


# =====================================================================


if __name__ == '__main__':
    # Download the data set
    print("Downloading data")
    frame = download_data()
    frame_sub = download_submission_data()

    # Process data into feature and label arrays
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test, X_Sub = get_features_and_labels(frame, frame_sub)
    # Evaluate multiple classifiers on the data
    print("Evaluating classifiers")
    results = list(evaluate_classifier(X_train, X_test, y_train, y_test, X_Sub))

    # Display the results
    print("Plotting the results")
    plot(results)

