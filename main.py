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
    INIT_SIMILARITY = -50
    q.put((INIT_SIMILARITY, 'ילד'))
    q.put((INIT_SIMILARITY, 'בגד'))
    q.put((INIT_SIMILARITY, 'חומר'))
    q.put((INIT_SIMILARITY, 'נוזל'))
    q.put((INIT_SIMILARITY, 'תפקיד'))
    q.put((INIT_SIMILARITY, 'מכשיר'))
    q.put((INIT_SIMILARITY, 'מקרר'))
    q.put((INIT_SIMILARITY, 'ילדה'))
    q.put((INIT_SIMILARITY, 'רגש'))
    q.put((INIT_SIMILARITY, 'תחושה'))
    q.put((INIT_SIMILARITY, 'חצאית'))
    q.put((INIT_SIMILARITY, 'שיער'))
    q.put((INIT_SIMILARITY, 'פנים'))
    q.put((INIT_SIMILARITY, 'מטוס'))
    q.put((INIT_SIMILARITY, 'מנוע'))

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

            topn = max(4, distance)
            topn = int(topn ** 0.5)
            for similar, _ in model.wv.most_similar(word, topn=topn):
                if similar not in seen:
                    q.put((-similarity, similar))
                    seen.add(similar)

            time.sleep(1)
