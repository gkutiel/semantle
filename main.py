from base64 import encode
import pandas as pd
import requests
import time
import json

from tqdm import tqdm
from itertools import product


def get(word):
    url = "https://semantle-he.herokuapp.com/api/distance?word=" + word
    r = requests.get(url).json()
    assert r is not None
    r['word'] = word
    return r


if __name__ == '__main__':
    from gensim.models import Word2Vec
    from queue import PriorityQueue

    model = Word2Vec.load('model.mdl')
    seen = set()
    q = PriorityQueue()
    q.put((50, 'ילד'))
    q.put((50, 'בגד'))
    q.put((50, 'חומר'))
    q.put((50, 'נוזל'))
    q.put((50, 'תפקיד'))
    q.put((50, 'מכשיר'))
    q.put((50, 'מקרר'))
    q.put((50, 'ילדה'))
    q.put((50, 'רגש'))
    q.put((50, 'תחושה'))
    q.put((50, 'חצאית'))
    q.put((50, 'שיער'))
    q.put((50, 'פנים'))
    q.put((50, 'מטוס'))
    q.put((50, 'מנוע'))

    with open('words.json', 'w', encoding='utf-8') as f:
        while q:
            p, word = q.get()
            seen.add(word)

            r = get(word)

            similarity = r['similarity']
            if not similarity:
                continue

            print(json.dumps(r), file=f)

            distance = r['distance']
            if distance > 700:
                print(f'[{q.qsize()}]', distance, r['word'])

            for similar, _ in model.wv.most_similar(word, topn=3):
                if similar not in seen:
                    q.put((100 - similarity, similar))
                    seen.add(similar)

            time.sleep(1)
