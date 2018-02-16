'''
Created on Feb 16, 2018

@author: fsteeg
'''

import json
import csv
import pickle
import requests
from pathlib import Path
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.classification import accuracy_score


bulk_file = 'nwbib-subjects-bulk.jsonl'
url = 'http://lobid.org/resources/search'
params = {
    'q': 'rheinland',  # for testing: use small-ish set
    'nested': 'subject:subject.source.id:"http://purl.org/lobid/nwbib"',
    'format': 'bulk'
}
saved_classifier_file = 'nwbib-subjects-classifier.pkl'
stop_word_url = 'https://raw.githubusercontent.com/solariz/german_stopwords/master/german_stopwords_plain.txt'
classifier = LinearSVC()

output_file = 'nwbib-subjects-bulk-predict.csv'


def main():

    vectorizer = HashingVectorizer(n_features=2 ** 18, stop_words=stop_words())

    create_bulk_data()
    hbzIds, subjects, texts = load_from_jsonl(bulk_file)

    test_set_size = len(subjects) // 100
    Y_train, X_train_texts = subjects[test_set_size:], texts[test_set_size:]
    ids_test, Y_test, X_test_texts = \
        hbzIds[:test_set_size], subjects[:test_set_size], texts[:test_set_size]

    print('{} training docs, {} testing docs'.format(len(Y_train), len(Y_test)))
    print_info(Y_train[0], X_train_texts[0], vectorizer)

    X_train = vectorizer.transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    classifier = create_classifier(X_train, Y_train)

    prediction, _score = predict(X_test, Y_test, classifier)

    data = [(hbzId, prediction[i]) for (i, hbzId) in enumerate(ids_test)]
    write_to_csv(output_file, data)


def stop_words():
    response = requests.get(
        url=stop_word_url)
    return [line.strip() for line in response.text.splitlines()
            if not line.startswith(';')]


def create_bulk_data():
    if not Path(bulk_file).exists():
        print('Getting bulk data...')
        response = requests.get(url=url, params=params)
        with open(bulk_file, 'w') as f:
            f.write(response.text)
    else:
        print('Using local bulk data in {}...'.format(bulk_file))


def create_classifier(X_train, Y_train):
    result = None
    if Path(saved_classifier_file).exists():
        print('Loading trained classifier...')
        with open(saved_classifier_file, 'rb') as c:
            result = pickle.load(c)
    else:
        print('Training classifier...')
        result = classifier.fit(X_train, Y_train)
        with open(saved_classifier_file, 'wb') as c:
            pickle.dump(result, c)
    return result


def print_info(subject, text, vectorizer):
    # See http://scikit-learn.org/stable/modules/feature_extraction.html
    print('Using vectorizer: {}'.format(vectorizer))
    analyzer = vectorizer.build_analyzer()
    vector = vectorizer.transform([text])[0]
    print('{}, {}'.format(subject, analyzer(text)))
    print(vector)


def predict(X_test, Y_test, classifier):
    print('Predicting...')
    Y_pred = classifier.predict(X_test)
    score = accuracy_score(Y_test, Y_pred)
    print('{:1.4f} classification accuracy for {}'.format(
        score, classifier))
    return Y_pred, score


def load_from_jsonl(jsonl):
    with open(jsonl, "r") as f:
        hbzIds = []
        subjects = []
        texts = []
        for line in f.readlines():
            entry = json.loads(line)
            hbzId = entry['hbzId']
            subject = first_nwbib_subject(entry)
            title = entry.get('title', '')
            sub = entry.get('otherTitleInformation', None)
            corp = entry.get('corporateBodyForTitle', None)
            vals = [title, sub[0] if sub else '', corp[0] if corp else '']
            doc = ' '.join(vals).strip()
            hbzIds.append(hbzId)
            subjects.append(subject)
            texts.append(doc)
    return (hbzIds, subjects, texts)


def write_to_csv(name, data):
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(
            csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['hbzId', 'subject'])
        for (hbzId, subject) in data:
            writer.writerow([hbzId, subject])


def first_nwbib_subject(entry):
    for subject in entry.get('subject', []):
        source = subject.get('source', None)
        if source and source.get('id', None) == 'http://purl.org/lobid/nwbib':
            return subject['id'].split('#')[1]
    return 'NULL'


main()
