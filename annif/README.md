# Annif workshop SWIB19, experiments with /nwbib data

See [https://github.com/NatLibFi/Annif-tutorial](https://github.com/NatLibFi/Annif-tutorial)

Basic idea:

```
wget https://github.com/hbz/lobid-vocabs/raw/master/nwbib/nwbib.ttl
annif loadvoc tfidf-nwbib nwbib.ttl
python3 nwbib_subjects_load_annif.py
annif train tfidf-nwbib nwbib-subjects-train-annif.tsv
```