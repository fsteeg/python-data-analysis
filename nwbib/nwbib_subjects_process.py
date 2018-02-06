'''
Created on Feb 7, 2018

@author: fsteeg

Process the data stored locally in CSV files.
'''

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import islice


def main():
    corpus = set()

    # Load the data from the CSV files:
    train = load_from_csv('nwbib-subjects-train.csv', corpus)
    test = load_from_csv('nwbib-subjects-test.csv', corpus)
    missing = load_from_csv('nwbib-subjects-missing.csv', corpus)

    # Build a dictionary from all documents in all sets:
    vectorizer = create_dictionary(corpus)
    print(list(islice(vectorizer.vocabulary_.items(), 3)))

    # Take a look at what we have:
    for doc in list(islice(train, 3)):
        hbzId, subject, text = doc
        analyzer = vectorizer.build_analyzer()
        print('{}, {}, {}'.format(hbzId, subject, analyzer(text)))

    # Create tf-idf vectors for each document based on the dictionary:
    train = transform(train, vectorizer)
    test = transform(test, vectorizer)
    missing = transform(missing, vectorizer)

    # Take a look at what we have:
    for doc in train[:3]:
        hbzId, subject, matrix = doc
        print('{}, {}, {} values:'.format(hbzId, subject, matrix.nnz))
        print(matrix)


def load_from_csv(f, corpus):
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row.
        result = set()
        for row in reader:
            hbzId, subject, title, otherTitleInformation, corporateBodyForTitle = row[
                0], row[1], row[2], row[3], row[4]
            doc = ','.join(
                [v for v in [title, otherTitleInformation, corporateBodyForTitle]
                 if v != 'NULL'])
            corpus.add(doc)
            result.add((hbzId, subject, doc))
        return result


def create_dictionary(corpus):
    # http://scikit-learn.org/stable/modules/feature_extraction.html
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(corpus)
    return vectorizer


def transform(data, vectorizer):
    result = []
    for doc in data:
        hbzId, subject, text = doc
        X = vectorizer.transform([text])
        result.append((hbzId, subject, X))
    return result


main()
