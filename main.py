import requests
import time
import json
from tqdm import tqdm


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
    INIT_SIMILARITY = -950

    def init_q(words):
        for word in words:
            q.put((INIT_SIMILARITY, word))

    init_q([
        'אמבטיה',
        'בגד',
        'בובה',
        'גג',
        'חדר',
        'חומר',
        'חצאית',
        'ילד',
        'ילדה',
        'מחשב',
        'מיטה',
        'מטוס',
        'מכשיר',
        'מנוע',
        'מקרר',
        'נוזל',
        'ספר',
        'עיניים',
        'עצמי',
        'פנים',
        'פעולה',
        'קציר',
        'רגש',
        'רהיט',
        'רצפה',
        'שולחן',
        'שעון',
        'שיער',
        'תחושה',
        'תפקיד',
    ])

    best_distance = -1
    best_word = None
    bar = tqdm()
    with open('words.json', 'w', encoding='utf-8') as f:
        while q:
            p, word = q.get()

            if word in seen:
                continue

            seen.add(word)
            r = get(word)

            if not r['similarity']:
                continue

            print(json.dumps(r), file=f)

            distance = r['distance']

            if distance == 1_000:
                print('*' * 20)
                print('found', word)
                print('*' * 20)
                break

            if distance > best_distance:
                best_distance = distance
                best_word = word

            bar.update(len(seen))
            bar.set_description(f'{best_word} {best_distance}')

            topn = int(max(1, distance))
            topn = topn ** 1.6
            topn = int(topn / 630)
            topn = max(1, topn)

            for similar, _ in model.wv.most_similar(word, topn=topn):
                if similar not in seen:
                    q.put((-distance, similar))

            time.sleep(1)
