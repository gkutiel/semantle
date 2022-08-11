import os
import heapq as hq
import requests
import time
import json
from tqdm import tqdm
from gensim.models import Word2Vec
from datetime import datetime as dt


def get(word):
    url = "https://semantle-he.herokuapp.com/api/distance?word=" + word
    r = requests.get(url).json()
    assert r is not None
    r['word'] = word
    return r


def notify(title, msg):
    os.system(f'''
        osascript -e 'display notification "{msg}" with title "{title}"'
        ''')


if __name__ == '__main__':
    model = Word2Vec.load('model.mdl')
    seen = set()
    q = []

    with open('seed.txt') as f:
        for word in f.read().splitlines():
            hq.heappush(q, (-50, word))

    best_similarity = -1
    best_word = None
    bar = tqdm()
    date = dt.strftime(dt.now(), '%Y-%m-%d')
    with open(f'{date}.json', 'w', encoding='utf-8') as f:
        while q:
            p, word = hq.heappop(q)

            if word in seen:
                continue

            seen.add(word)
            r = get(word)
            time.sleep(1)

            similarity = r['similarity']
            if not similarity:
                continue

            print(json.dumps(r), file=f)

            if similarity > best_similarity:
                notify(word, similarity)
                best_similarity = similarity
                best_word = word

            bar.update()
            bar.set_description(f'{best_word} {best_similarity}')

            for similar, _ in model.wv.most_similar(word, topn=30):
                hq.heappush(q, (-similarity, similar))

            if r['distance'] == 1_000:
                print('\n' * 3)
                print('*' * 20)
                print('found', word)
                print('*' * 20)
                notify(word, similarity)
                break
