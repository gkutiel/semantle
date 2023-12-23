import heapq as hq
import json
import os
import time
from datetime import datetime as dt
from sys import platform

import requests
from gensim.models import Word2Vec
from tqdm import tqdm


def get(word):
    url = "https://semantle.ishefi.com/api/distance?word=" + word
    r = requests.get(url).json()
    assert r is not None
    r['word'] = word
    return r


def notify(title, msg):
    if 'linux' in platform:
        os.system(f'notify-send " {title}" " {msg}"')
    elif 'win' in platform:
        pass
    else:
        os.system(
            f'''osascript -e 'display notification "{msg}" with title "{title}"' ''')


if __name__ == '__main__':
    model = Word2Vec.load('model.mdl')
    seen = set()
    q = []

    with open('seed.txt') as f:
        for word in f.read().splitlines():
            hq.heappush(q, (-70, word))

    best_similarity = -1
    best_word = None
    bar = tqdm(unit=' it ')
    date = dt.strftime(dt.now(), '%Y-%m-%d')
    with open('last.json', 'w', encoding='utf-8') as last:
        with open(f'{date}.json', 'w', encoding='utf-8') as f:
            while q:
                try:
                    p, word = hq.heappop(q)

                    if word in seen:
                        continue

                    seen.add(word)
                    r = get(word)
                    time.sleep(1)

                    similarity = r['similarity']
                    if not similarity:
                        continue

                    distance = r['distance']

                    print(json.dumps(r), file=f)
                    print(json.dumps(r), file=last)

                    if similarity > best_similarity:
                        notify(word, distance)
                        best_similarity = similarity
                        best_word = word
                        bar.set_description(f'{best_word} {distance}')

                    bar.update()

                    for similar, _ in model.wv.most_similar(word, topn=30):
                        hq.heappush(q, (-similarity, similar))

                    if r['distance'] == 1_000:
                        print('\n' * 3)
                        print('*' * 20)
                        print('found', word)
                        print('*' * 20)
                        break
                except:
                    pass
