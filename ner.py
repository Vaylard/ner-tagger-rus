import glob
import itertools
import pymorphy2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.layers import (
    Input, Embedding,
    Conv1D, MaxPooling1D, Flatten,
    Concatenate,
    Bidirectional, LSTM,
    TimeDistributed, Dense
)

def mark_bi_tags(bio_tags, token_id_to_ind, token_ids, tag):
    first_loc = token_id_to_ind[token_ids[0]]
    bio_tags[first_loc[0]][first_loc[1]] = 'B_' + tag
    for inside_token in token_ids[1:]:
        loc = token_id_to_ind[inside_token]
        bio_tags[loc[0]][loc[1]] = 'I_' + tag

def make_corpus_from_files(directory):
    corpus_sentences = []
    corpus_tags = []

    token_files = glob.glob(directory + '/*.tokens')
    for token_file in token_files:
        with open(token_file) as f:
            token_lists = [s.split('\n') for s in f.read().split('\n\n')][:-1]
            sentences_with_ids = [[(t.split()[0], t.split()[3]) for t in tl] for tl in token_lists]
            token_id_to_ind = dict()
            for s_ind in range(len(sentences_with_ids)):
                for w_ind in range(len(sentences_with_ids[s_ind])):
                    token_id_to_ind[sentences_with_ids[s_ind][w_ind][0]] = (s_ind, w_ind)
    
        with open(token_file[:-7] + '.spans') as f:
            bio_tags = [['O' for w in s] for s in sentences_with_ids]
            spans = [l.split() for l in f.readlines()]
            name_span_buffer = []
            for span in spans:
                span_type = span[1]
                token_ids = span[7: 7 + int(span[5])]

                if span_type in ['name', 'surname', 'patronymic']:
                    if len(name_span_buffer) == 0:
                        name_span_buffer = token_ids
                    elif int(token_ids[0]) > int(name_span_buffer[-1]) + 1:
                        mark_bi_tags(bio_tags, token_id_to_ind, name_span_buffer, 'PER')
                        name_span_buffer = token_ids
                    else:
                        name_span_buffer += token_ids
                elif len(name_span_buffer) > 0:
                    mark_bi_tags(bio_tags, token_id_to_ind, name_span_buffer, 'PER')
                    name_span_buffer = []
            
                if span_type == 'org_name':
                    mark_bi_tags(bio_tags, token_id_to_ind, token_ids, 'ORG')
                elif span_type == 'loc_name':
                    mark_bi_tags(bio_tags, token_id_to_ind, token_ids, 'LOC')
        if len(name_span_buffer) > 0:
            mark_bi_tags(bio_tags, token_id_to_ind, name_span_buffer, 'PER')
    
        corpus_sentences += [[t[1] for t in s] for s in sentences_with_ids]
        corpus_tags += bio_tags
    
    return (corpus_sentences, corpus_tags)

devset_sentences, devset_tags = make_corpus_from_files('./ner_corpus/devset')
testset_sentences, testset_tags = make_corpus_from_files('./ner_corpus/testset')

corpus_dev_s = testset_sentences + devset_sentences[:700]
corpus_dev_t = testset_tags + devset_tags[:700]
corpus_test_s = devset_sentences[700:]
corpus_test_t = devset_tags[700:]

print('Sentences (dev):', len(corpus_dev_s))
print('Tokens (dev):', sum(len(s) for s in corpus_dev_s))
print('Sentences (test):', len(corpus_test_s))
print('Tokens (test):', sum(len(s) for s in corpus_test_s))

from gensim.models import KeyedVectors

# 3rd party embeddings trained on the entire text of Russian wikipedia
full_wiki_embs = KeyedVectors.load_word2vec_format('./embeddings/ruwiki_20180420_300d.txt', binary=False)

# "home-made" embeddings trained on a single category
part_wiki_embs = KeyedVectors.load('./embeddings/partial_wiki_100d.keyedvectors')

vocabulary = list(set(itertools.chain(*(corpus_dev_s + corpus_test_s))))
ner_tags = ['O', 'B_ORG', 'I_ORG', 'B_LOC', 'I_LOC', 'B_PER', 'I_PER']
ner_tags_index = {tag: index + 1 for index, tag in enumerate(ner_tags)}

morph = pymorphy2.MorphAnalyzer()
def normalize(word):
    return morph.parse(word)[0].normal_form
def is_punct(word):
    return 'PNCT' in morph.parse(word)[0].tag

def get_embedding(word, dim, emb_model):
    if word in emb_model:
        return emb_model[word]
    elif is_punct(word):
        return np.zeros((dim,))
    else:
        return np.zeros((dim,)) - 1.0

vocabulary_norm = list(set(normalize(w) for w in vocabulary)) + ['__UNK__']
vocab_index_norm = {word: index + 1 for index, word in enumerate(vocabulary_norm)}

emb_matrix_full = np.zeros((len(vocabulary_norm) + 1, 300))
for index, word in enumerate(vocabulary_norm):
    emb_matrix_full[index + 1] = get_embedding(word, 300, full_wiki_embs)

emb_matrix_part = np.zeros((len(vocabulary_norm) + 1, 100))
for index, word in enumerate(vocabulary_norm):
    emb_matrix_part[index + 1] = get_embedding(word, 100, part_wiki_embs)

def make_X_w2v(corpus):
    X = []
    for s in corpus:
        s_inds = []
        for w in s:
            lemma = normalize(w)
            if lemma in vocab_index_norm:
                s_inds.append(vocab_index_norm[lemma])
            else:
                s_inds.append(vocab_index_norm['__UNK__'])
        X.append(s_inds)
    X = pad_sequences(X, maxlen=100, padding='post')
    return np.array(X)

del full_wiki_embs
del part_wiki_embs

alphabet_cyr = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
alphabet_lat = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'
basic_symbols = list(alphabet_cyr.upper() + alphabet_cyr + alphabet_lat.upper() + alphabet_lat + digits)
extra_symbols = list(set(itertools.chain(*(list(w) for w in vocabulary))) - set(basic_symbols))
char_vocab = basic_symbols + extra_symbols + ['__UNK__']

def make_X_cnn(corpus):
    X = []
    for s in corpus:
        sent = np.zeros((100, 20))
        for i, w in enumerate(s):
            for j, c in enumerate(w):
                if j >= 20:
                    break
                if c in char_vocab:
                    c_ind = char_vocab.index(c)
                else:
                    c_ind = char_vocab.index('__UNK__')
                sent[i][j] = c_ind
        X.append(sent)
    return np.array(X)

def make_y(tag_sequences):
    y = []
    for s in tag_sequences:
        y.append([ner_tags_index[t] for t in s])
    y = to_categorical(pad_sequences(y, maxlen=100, padding='post'))
    return y

X_w2v_dev, X_cnn_dev, y_dev = make_X_w2v(corpus_dev_s), make_X_cnn(corpus_dev_s), make_y(corpus_dev_t)
X_w2v_test, X_cnn_test, y_test = make_X_w2v(corpus_test_s), make_X_cnn(corpus_test_s), make_y(corpus_test_t)

def make_model_with_embeddings(emb_matrix):
    inp_cnn = Input(shape=(100, 20))
    x = TimeDistributed(Embedding(len(char_vocab), 64, input_length=20))(inp_cnn)
    x = TimeDistributed(Conv1D(filters=128,
                               kernel_size=5,
                               padding='same',
                               activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=4, padding='valid'))(x)
    x = TimeDistributed(Conv1D(filters=64,
                               kernel_size=5,
                               padding='same',
                               activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=4, padding='valid'))(x)
    x = TimeDistributed(Flatten())(x)
    emb_cnn = TimeDistributed(Dense(64, activation='relu'))(x)

    inp_w2v = Input(shape=(100,))
    emb_w2v = Embedding(len(vocabulary_norm) + 1, emb_matrix.shape[1],
                        weights=[emb_matrix],
                        input_length=100,
                        trainable=False)(inp_w2v)

    feats = Concatenate()([emb_w2v, emb_cnn])
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(feats)
    x = LSTM(128, return_sequences=True, dropout=0.2)(x)
    out = TimeDistributed(Dense(len(ner_tags) + 1, activation='softmax'))(x)

    model = Model(inputs=[inp_w2v, inp_cnn], outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

ner_tagger_full = make_model_with_embeddings(emb_matrix_full)
print(ner_tagger_full.summary())

ner_tagger_part = make_model_with_embeddings(emb_matrix_part)
print(ner_tagger_part.summary())

import re

def get_ne_spans(tags):
    spans = []
    span_buffer = []
    for i, t in enumerate(tags):
        if re.fullmatch(r'^B_...$', t):
            if len(span_buffer) > 0:
                spans.append((span_buffer[0][0], span_buffer[-1][0], span_buffer[0][1][2:]))
                span_buffer = []
            span_buffer.append((i, t))
        elif re.fullmatch(r'^I_...$', t) and len(span_buffer) > 0 and span_buffer[0][1][2:] == t[2:]:
            span_buffer.append((i, t))
        elif len(span_buffer) > 0:
            spans.append((span_buffer[0][0], span_buffer[-1][0], span_buffer[0][1][2:]))
            span_buffer = []
    if len(span_buffer) > 0:
        spans.append((span_buffer[0][0], span_buffer[-1][0], span_buffer[0][1][2:]))
        span_buffer = []
    return spans

def calculate_f1_scores(y_tags_gold, y_tags_pred):
    tp_all, fp_all, fn_all = 0, 0, 0
    tp_tag = {'ORG': 0, 'LOC': 0, 'PER': 0}
    fp_tag = {'ORG': 0, 'LOC': 0, 'PER': 0}
    fn_tag = {'ORG': 0, 'LOC': 0, 'PER': 0}
    for i in range(len(y_tags_gold)):
        spans_gold = set(get_ne_spans(y_tags_gold[i]))
        spans_pred = set(get_ne_spans(y_tags_pred[i]))

        tp_all += len(spans_gold & spans_pred)
        fp_all += len(spans_pred - spans_gold)
        fn_all += len(spans_gold - spans_pred)
        for tag in tp_tag:
            spans_tag_gold = set(s for s in spans_gold if s[2] == tag)
            spans_tag_pred = set(s for s in spans_pred if s[2] == tag)
            tp_tag[tag] += len(spans_tag_gold & spans_tag_pred)
            fp_tag[tag] += len(spans_tag_gold - spans_tag_pred)
            fn_tag[tag] += len(spans_tag_gold - spans_tag_pred)

    f1_scores = dict()
    f1_scores['ALL'] = tp_all / (tp_all + 0.5*(fp_all + fn_all))
    for tag in tp_tag:
        f1_scores[tag] = tp_tag[tag] / (tp_tag[tag] + 0.5*(fp_tag[tag] + fn_tag[tag]))
    
    return f1_scores

def preds_to_tags(y_probs):
    y_tags = []
    for s in y_probs:
        sentence = []
        for w in s:
            tag_num = np.argmax(w)
            if tag_num > 0:
                sentence.append(ner_tags[tag_num - 1])
            else:
                sentence.append('PAD')
        y_tags.append(sentence)
    return y_tags

def plot_f1s(dev_f1s, test_f1s, num_epochs):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title('All tags')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.ylim([0.0, 1.0])
    plt.plot(range(1, num_epochs + 1), [x['ALL'] for x in dev_f1s], label='dev')
    plt.plot(range(1, num_epochs + 1), [x['ALL'] for x in test_f1s], label='test')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.title('Locations')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.ylim([0.0, 1.0])
    plt.plot(range(1, num_epochs + 1), [x['LOC'] for x in dev_f1s], label='dev')
    plt.plot(range(1, num_epochs + 1), [x['LOC'] for x in test_f1s], label='test')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.title('Organizations')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.ylim([0.0, 1.0])
    plt.plot(range(1, num_epochs + 1), [x['ORG'] for x in dev_f1s], label='dev')
    plt.plot(range(1, num_epochs + 1), [x['ORG'] for x in test_f1s], label='test')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.title('Personal names')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.ylim([0.0, 1.0])
    plt.plot(range(1, num_epochs + 1), [x['PER'] for x in dev_f1s], label='dev')
    plt.plot(range(1, num_epochs + 1), [x['PER'] for x in test_f1s], label='test')
    plt.legend()

    plt.show()

dev_f1s = []
test_f1s = []

for i in range(50):
    hist = ner_tagger_full.fit([X_w2v_dev, X_cnn_dev], y_dev, batch_size=256, epochs=1, verbose=0)
    if (i == 0 or i % 10 == 9):
        print(f'Epoch {i + 1} / 50')
        print(f"loss: {hist.history['loss'][0]}, acc: {hist.history['acc'][0]}")
    
    y_tags_dev = preds_to_tags(ner_tagger_full.predict([X_w2v_dev, X_cnn_dev]))
    y_tags_test = preds_to_tags(ner_tagger_full.predict([X_w2v_test, X_cnn_test]))
    dev_f1s.append(calculate_f1_scores(corpus_dev_t, y_tags_dev))
    test_f1s.append(calculate_f1_scores(corpus_test_t, y_tags_test))

plot_f1s(dev_f1s, test_f1s, 50)

dev_f1s = []
test_f1s = []

for i in range(50):
    hist = ner_tagger_part.fit([X_w2v_dev, X_cnn_dev], y_dev, batch_size=256, epochs=1, verbose=0)
    if (i == 0 or i % 10 == 9):
        print(f'Epoch {i + 1} / 50')
        print(f"loss: {hist.history['loss'][0]}, acc: {hist.history['acc'][0]}")
    
    y_tags_dev = preds_to_tags(ner_tagger_part.predict([X_w2v_dev, X_cnn_dev]))
    y_tags_test = preds_to_tags(ner_tagger_part.predict([X_w2v_test, X_cnn_test]))
    dev_f1s.append(calculate_f1_scores(corpus_dev_t, y_tags_dev))
    test_f1s.append(calculate_f1_scores(corpus_test_t, y_tags_test))

plot_f1s(dev_f1s, test_f1s, 50)
