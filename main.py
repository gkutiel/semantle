import heapq as hq
import json
import os
import time
from datetime import datetime as dt
from sys import platform

import requests
from gensim.models import Word2Vec
from tqdm import tqdm


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
    best_distance = -2
    best_word = None
    date = dt.strftime(dt.now(), '%Y-%m-%d')
    with open('last.json', 'w') as last:
        with open(f'{date}.json', 'w') as current:
            def add(word) -> int:
                if word in seen:
                    return -1

                try:
                    seen.add(word)
                    url = f'https://semantle.ishefi.com/api/distance?word={word}'
                    r = requests.get(url).json() | {'word': word}
                    dump(r, current, last)
                    hq.heappush(q, (-r['similarity'], word))
                    return r['distance']
                finally:
                    time.sleep(1)

            def add_all(words, desc):
                global best_distance, best_word
                words = tqdm(words, desc=desc)
                for word in words:
                    dist = add(word)
                    if dist > best_distance:
                        best_distance = dist
                        best_word = word

                    words.set_postfix(
                        distance=best_distance,
                        word=best_word)

                    if dist == 1000:
                        break

            with open('seed.txt', encoding='utf8') as seed:
                add_all(seed.read().splitlines(), desc='Loading seed')

            def words():
                while q:
                    _, word = hq.heappop(q)
                    for similar, _ in model.wv.most_similar(word, topn=30):
                        yield similar

            add_all(words(), desc='Searching')
