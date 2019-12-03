'''
Created in Novemeber 2019

@author: fsteeg

Load data from the Lobid API, store as CSV files for further processing.
'''

import csv
import json
import requests


url = 'http://lobid.org/resources/search'

train = {
    # try to approach full NWBib set: learn from a complete, spatially
    # restricted data set with somewhat even subject distribution, see
    # https://test.nwbib.de/search?q=hochsauerland ('Sachgebiete' facet)
    'q': 'hochsauerland',
    'nested': 'subject:subject.source.id:"https://nwbib.de/subjects"',
    'format': 'json',
    'size': '450'
}
test = {
    'q': 'hochsauerland',
    'nested': 'subject:subject.source.id:"https://nwbib.de/subjects"',
    'format': 'json',
    'size': '50',
    'from': '451'
}
missing = {
    'filter': 'inCollection.id:"http://lobid.org/resources/HT014176012#!"',
    'q': 'NOT _exists_:subject',
    'format': 'json',
    'size': '2000'
}


def main():
    load_to_csv(train, 'nwbib-subjects-train-annif.tsv')
    load_to_csv(test, 'nwbib-subjects-test.csv')
    load_to_csv(missing, 'nwbib-subjects-missing.csv')


def load_to_csv(params, name):
    resp = requests.get(url=url, params=params)
    data = json.loads(resp.text)
    print("Got {} entries for {} (total: {})".format(
        len(data['member']), name, data['totalItems']))
    write_csv(name, data)


def write_csv(name, data):
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(
            csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(
        #   ['hbzId', 'subject', 'title', 'otherTitleInformation', 'corporateBodyForTitle'])
        for entry in data['member']:
            hbzId = entry['hbzId']
            subject = first_nwbib_subject(entry)
            title = entry.get('title', 'NULL')
            sub = entry.get('otherTitleInformation', None)
            corp = entry.get('corporateBodyForTitle', None)
            writer.writerow(
                #[hbzId, subject, title, sub[0] if sub else 'NULL', corp[0] if corp else 'NULL'])
                [title+' ' +sub[0] if sub else ''+ ' ' + corp[0] if corp else '', '<http://purl.org/lobid/nwbib#s'+subject[1:]+'>'])


def first_nwbib_subject(entry):
    for subject in entry.get('subject', []):
        source = subject.get('source', None)
        if source and source.get('id', None) == 'https://nwbib.de/subjects':
            return subject['id'].split('#')[1]
    return 'NULL'


main()
