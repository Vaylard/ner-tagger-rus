import requests
from bs4 import BeautifulSoup
import nltk
import pymorphy2
from gensim.models import Word2Vec

def api_req(**kwargs):
    API_URL='https://ru.wikipedia.org/w/api.php'
    params = '&'.join(f'{k}={v}' for k, v in kwargs.items())
    return requests.get(API_URL + '?' + params + '&format=json').json()

def category_bfs(root_category_title):
    pages = set()
    visited_categories = set([root_category_title])

    r = api_req(action='query',
                list='categorymembers',
                cmtitle=root_category_title,
                cmprop='title|type',
                cmlimit='500')
    objs = r['query']['categorymembers']
    pages.update([o['title'] for o in objs if o['type'] == 'page'])
    queue = [o['title'] for o in objs if o['type'] == 'subcat']

    while len(queue) > 0:
        visiting = queue
        queue = []
        for subcat in visiting:
            visited_categories.add(subcat)
            r = api_req(action='query',
                        list='categorymembers',
                        cmtitle=subcat,
                        cmprop='title|type',
                        cmlimit='500')
            objs = r['query']['categorymembers']
            pages.update([o['title'] for o in objs if o['type'] == 'page'])
            if len(pages) > 10000:
                queue = []
                break
            queue += [o['title'] for o in objs if o['type'] == 'subcat' and o['title'] not in visited_categories]
    
    return pages

pages = category_bfs('Категория:История_XX_века')
print(len(pages))

nltk.download('punkt')
morph = pymorphy2.MorphAnalyzer()
def normalize(word):
    return morph.parse(word.lower())[0].normal_form

corpus = []
for p in pages:
    r = api_req(action='parse',
                page='_'.join(p.split()),
                prop='text',
                formatversion=2)
    content_html = r['parse']['text']
    soup = BeautifulSoup(content_html, 'html.parser')
    relevant_text = '\n'.join([html_p.get_text() for html_p in soup.find_all('p')])
    sentences = nltk.sent_tokenize(relevant_text, language='russian')
    for s in sentences:
        words = nltk.word_tokenize(s, language='russian')
        words = [normalize(w) for w in words if w.isalpha()]
        corpus.append(words)

print('Sentences:', len(corpus))
print('Tokens:', sum(len(s) for s in corpus))

model = Word2Vec(sentences=corpus, size=100, window=5, min_count=2, workers=4)
model.train(sentences=corpus, total_examples=len(corpus), epochs=10)

model.wv.save('./embeddings/partial_wiki_100d.keyedvectors')