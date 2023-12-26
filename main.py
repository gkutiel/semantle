import heapq as hq
import json
import os
import time
import traceback
from datetime import datetime as dt
from sys import platform
from urllib import parse

import requests
from gensim.models import Word2Vec
from tqdm import tqdm


def get(word):
    time.sleep(1)
    url = f'https://semantle.ishefi.com/api/distance?word={word}'
    r = requests.get(url).json()
    assert r is not None
    r['word'] = word
    return r


def notify(title, msg):
    if 'linux' in platform:
        os.system(f'notify-send " {title}" " {msg}"')


def dump(obj, *files):
    for file in files:
        print(json.dumps(obj, ensure_ascii=False), file=file)


if __name__ == '__main__':
    model = Word2Vec.load('model.mdl')
    seen = set()
    q = []

    with open('seed.txt', encoding='utf8') as f:
        for word in tqdm(f.read().splitlines(), desc='Loading seed'):
            hq.heappush(q, (-60, word))

    errors = 0
    best_similarity = -1
    best_distance = 0
    best_word = None
    bar = tqdm(unit='it', desc='Searching')
    date = dt.strftime(dt.now(), '%Y-%m-%d')
    with open('last.json', 'w', encoding='utf-8') as last:
        with open(f'{date}.json', 'w', encoding='utf-8') as f:
            while q and best_distance < 1_000:
                try:
                    p, word = hq.heappop(q)

                    if word in seen:
                        continue

                    seen.add(word)

                    r = get(word)

                    similarity = r['similarity']
                    distance = r['distance']

                    if similarity > best_similarity:
                        # notify(word, distance)
                        best_similarity = similarity
                        best_distance = distance
                        best_word = word

                    dump(r, f, last)

                    for similar, _ in model.wv.most_similar(word, topn=10):
                        hq.heappush(q, (-similarity, similar))

                    bar.set_postfix(
                        errors=errors,
                        similarity=best_similarity,
                        distance=best_distance,
                        word=best_word)

                    bar.update()

                except Exception:
                    errors += 1
