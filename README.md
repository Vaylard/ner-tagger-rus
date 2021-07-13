# ner-tagger-rus

### Launching the scripts
The scripts must be run in the following order:
1. `wiki_word2vec.py` (generate `embeddings/partial_wiki_100d.keyedvectors`)
2. `ner.py` (train the NER model)

In order to run, the scripts require the following directories and files:

`wiki_word2vec.py`:
- `embeddings/`

`ner.py`:
- `ner_corpus/`
- `embeddings/`
- `ner_corpus/devset/`, `ner_corpus/testset/` \
Available at: https://github.com/dialogue-evaluation/factRuEval-2016
- `embeddings/ruwiki_20180420_300d.txt` \
Available at: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

### Dependencies
Recommended Python version: `3.7.10`

`tensorflow==2.5.0` \
`numpy==1.19.5` \
`pymorphy2==0.9.1` \
`nltk==3.2.5` \
`gensim==3.6.0` \
`bs4==0.0.1` \
`matplotlib==3.2.2` \
`requests==2.23.0`
