'''
Created on Feb 7, 2018

@author: fsteeg

Process the data stored locally in CSV files.
'''

import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.classification import accuracy_score


def main():

    # Set up the experiment:

    train = 'nwbib-subjects-train.csv'
    test = 'nwbib-subjects-test.csv'
    output = 'nwbib-subjects-predict.csv'

    vectorizers = [
        CountVectorizer(binary=True),
        CountVectorizer(ngram_range=(1, 2)),
        TfidfVectorizer(),
        TfidfVectorizer(ngram_range=(1, 2))]

    classifiers = [
        GaussianNB(),
        BernoulliNB(),
        LogisticRegression(),
        LinearSVC(),
        DecisionTreeClassifier(max_depth=5)]

    run_experiment(train, test, output, vectorizers, classifiers)


def run_experiment(train, test, output, vectorizers, classifiers):

    best_score = 0
    best_prediction = []
    best_info = None

    # Load the data from the CSV files:
    corpus = set()
    ids_train, Y_train, X_train_texts = load_from_csv(train, corpus)
    ids_test, Y_test, X_test_texts = load_from_csv(test, corpus)

    for vectorizer in vectorizers:

        # Build a dictionary from all documents using different vectorizers:
        vectorizer = vectorizer.fit(corpus)
        print_info(ids_train[0], Y_train[0], X_train_texts[0], vectorizer)

        # Transform texts to feature vectors based on the dictionary:
        X_train = vectorizer.transform(X_train_texts).toarray()
        X_test = vectorizer.transform(X_test_texts).toarray()

        for classifier in classifiers:

            # Predict classes for the test set using different classifiers:
            prediction, score = predict(
                X_train, Y_train, X_test, Y_test, classifier)

            # Track the best result so far:
            if score > best_score:
                best_score, best_prediction = score, prediction
                best_info = '{}\n{}'.format(vectorizer, classifier)

    # Write best prediction to a CSV file:
    print('\nBest score: {:1.4f}, writing to {}, using:'.format(
        best_score, output))
    print(best_info)
    data = [(hbzId, best_prediction[i]) for (i, hbzId) in enumerate(ids_test)]
    write_to_csv(output, data)


def predict(X_train, Y_train, X_test, Y_test, classifier):
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    score = accuracy_score(Y_test, Y_pred)
    print('{:1.4f} classification accuracy for {}'.format(
        score, type(classifier)))
    return Y_pred, score


def load_from_csv(f, corpus):
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row.
        hbzIds = []
        X = []
        Y = []
        for row in reader:
            hbzId, subject, title, otherTitleInformation, corporateBodyForTitle = row[
                0], row[1], row[2], row[3], row[4]
            doc = ','.join(
                [v for v in [title, otherTitleInformation, corporateBodyForTitle]
                 if v != 'NULL'])
            corpus.add(doc)
            hbzIds.append(hbzId)
            Y.append(subject)
            X.append(doc)
        return hbzIds, Y, X


def write_to_csv(name, data):
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(
            csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['hbzId', 'subject'])
        for (hbzId, subject) in data:
            writer.writerow([hbzId, subject])


def print_info(hbzId, subject, text, vectorizer):
    # See http://scikit-learn.org/stable/modules/feature_extraction.html
    print()
    print('Using vectorizer: {}'.format(vectorizer))
    analyzer = vectorizer.build_analyzer()
    vector = vectorizer.transform([text])[0]
    print('{}, {}, {}'.format(hbzId, subject, analyzer(text)))
    print(vector)


main()
