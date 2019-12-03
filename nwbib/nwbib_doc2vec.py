'''
Created on Mar 5, 2018

See also: https://radimrehurek.com/gensim/models/doc2vec.html

@author: fsteeg
'''

import collections
import gensim
import json
import random
import smart_open


def main():

    # see nwbib_subjects_bulk.py for getting the data
    corpus = list(load_from_jsonl('nwbib-subjects-bulk.jsonl'))

    test_set_size = len(corpus) // 100
    train_corpus = corpus[test_set_size:]
    test_corpus = corpus[:test_set_size]

    print('Training: {} docs, testing: {} docs'.format(
        len(train_corpus), len(test_corpus)))
    print(train_corpus[:2])
    print(test_corpus[:2])

    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=20, window=5, min_count=2, epochs=20, workers=4)
    model.build_vocab(train_corpus + test_corpus)
    model.train(train_corpus, total_examples=model.corpus_count,
                epochs=model.epochs)
    model.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)
    print("Model: {}".format(model))

    _vector = model.infer_vector(
        ['länderübergreifendes', 'beweidungskonzept', 'mit', 'rhönschafen', 'realisiert'])

    counter = collections.Counter(compute_ranks(model, train_corpus))
    print('\nRanking of training docs in their own most_similar docs (goal: most in 0 and 1):')
    print(counter.most_common(10))

    # Test with random document from the training set
    doc_id = random.randint(0, len(train_corpus) - 1)
    find_similar(model, doc_id, train_corpus, train_corpus)

    # Test with random document from the testing set
    doc_id = random.randint(0, len(test_corpus) - 1)
    find_similar(model, doc_id, test_corpus, train_corpus)


def find_similar(model, doc_id, corpus1, corpus2):
    inferred_vector = model.infer_vector(corpus1[doc_id].words)
    similar = model.docvecs.most_similar(
        positive=[inferred_vector], topn=len(model.docvecs))
    self_sim = 1.0 - model.docvecs.distance(
        corpus1[doc_id].tags[0], corpus1[doc_id].tags[0])
    print('\nTrain Document ({} ~ {} = {:1.2f}): «{}»'.format(
        doc_id, doc_id, self_sim, ' '.join(corpus1[doc_id].words)))
    print('Most similar ({} ~ {} = {:1.2f}): «{}»'.format(
        doc_id, similar[0][0], similar[0][1], ' '.join(corpus2[similar[0][0]].words)))


def compute_ranks(model, train_corpus):
    ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        similar = model.docvecs.most_similar(
            [inferred_vector], topn=len(model.docvecs))
        rank = [d_id for d_id, _sim in similar].index(doc_id)
        ranks.append(rank)
    return ranks


def load_from_jsonl(jsonl):
    with smart_open.smart_open(jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            title = entry.get('title', '')
            sub = entry.get('otherTitleInformation', None)
            corp = entry.get('corporateBodyForTitle', None)
            vals = [title, sub[0] if sub else '', corp[0] if corp else '']
            doc = ' '.join(vals).strip()
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc, max_len=20), [i])


main()
