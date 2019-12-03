'''
Created on Feb 23, 2018

@author: fsteeg
'''

import json
import requests
from pprint import pprint
from time import time
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging
from sklearn.model_selection import ShuffleSplit


def main():

    # print progress to standard out
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # the input data, get it using nwbib_subjects_bulk.py
    bulk_file = 'nwbib-subjects-bulk.jsonl'
    _hbzIds, Y_train, X_train = load_from_jsonl(bulk_file)

    # configure the cross-validation count and split size
    validator = ShuffleSplit(n_splits=5, test_size=0.01)

    pipeline = Pipeline([
        ('vect', HashingVectorizer()),
        ('clf', LinearSVC()),
    ])

    parameters = {
        'vect__binary': [False, True],
        'vect__stop_words': [None,
                             # stop_words('german_stopwords_full.txt'),
                             stop_words('german_stopwords_plain.txt')],
        'vect__n_features': [2 ** 18], #, 2 ** 19
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__max_iter': [10] #, 100
    }

    jobs = 4

    # multiprocessing requires the fork to happen in a protected block:
    if __name__ == "__main__":

        # find the best parameters for feature extraction and classification
        grid_search = GridSearchCV(
            pipeline, parameters, n_jobs=jobs, verbose=1, cv=validator)

        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)
        print('Using {} concurrent jobs (more jobs = lower runtime, but higher memory usage)'.format(
            jobs))
        start = time()

        grid_search.fit(X_train, Y_train)

        print("done in %0.3fs" % (time() - start))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))


def stop_words(file):
    response = requests.get(
        url='https://raw.githubusercontent.com/solariz/german_stopwords/master/' + file)
    return [line.strip() for line in response.text.splitlines()
            if not line.startswith(';')]


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


def first_nwbib_subject(entry):
    for subject in entry.get('subject', []):
        source = subject.get('source', None)
        if source and source.get('id', None) == 'https://nwbib.de/subjects':
            return subject['id'].split('#')[1]
    return 'NULL'


main()
