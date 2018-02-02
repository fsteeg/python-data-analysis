'''
Created on Feb 2, 2018

@author: fsteeg
'''

import json
import requests
import csv


url = 'http://lobid.org/resources/search'

train = {
    'nested': 'subject:subject.source.id:"http://purl.org/lobid/nwbib"',
    'format': 'json',
    'size': '2000'
}
test = {
    'nested': 'subject:subject.source.id:"http://purl.org/lobid/nwbib"',
    'format': 'json',
    'size': '2000',
    'from': '2001'
}
missing = {
    'filter': 'inCollection.id:"http://lobid.org/resources/HT014176012#!"',
    'q': 'NOT _exists_:subject',
    'format': 'json',
    'size': '2000'
}


def main():
    load_to_csv(train, 'nwbib-subjects-train.csv')
    load_to_csv(test, 'nwbib-subjects-test.csv')
    load_to_csv(missing, 'nwbib-subjects-missing.csv')


def load_to_csv(params, name):
    resp = requests.get(url=url, params=params)
    data = json.loads(resp.text)
    print("Got {} entries for {}".format(len(data['member']), name))
    write_csv(name, data)


def write_csv(name, data):
    with open(name, 'w', newline='') as csvfile:
        missing_writer = csv.writer(
            csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        missing_writer.writerow(
            ['hbzId', 'subject', 'title', 'otherTitleInformation', 'corporateBodyForTitle'])
        for entry in data['member']:
            hbzId = entry['hbzId']
            subject = first_nwbib_subject(entry)
            title = entry.get('title', 'NULL')
            sub = entry.get('otherTitleInformation', None)
            corp = entry.get('corporateBodyForTitle', None)
            missing_writer.writerow(
                [hbzId, subject, title, sub[0] if sub else 'NULL', corp[0] if corp else 'NULL'])


def first_nwbib_subject(entry):
    for subject in entry.get('subject', []):
        source = subject.get('source', None)
        if source and source.get('id', None) == 'http://purl.org/lobid/nwbib':
            return subject['id'].split('#')[1]
    return 'NULL'


main()
