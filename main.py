import requests
import time
import json


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
            seen.add(word)
            q.put((INIT_SIMILARITY, word))

    init_q([
        'אמבטיה',
        'בגד',
        'בובה',
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
        'שעון',
        'שיער',
        'תחושה',
        'תפקיד',
    ])

    best_distance = -1
    words = []
    with open('words.json', 'w', encoding='utf-8') as f:
        while q:
            p, word = q.get()
            r = get(word)

            if not r['similarity']:
                continue

            print(json.dumps(r), file=f)

            distance = r['distance']
            best_distance = max(best_distance, distance)

            if distance == 1_000:
                break

            words.append((distance, word))
            print()
            for dis, word in sorted(words)[-10:]:
                print(f'{dis:<5} {word}')

            topn = max(4, distance)
            topn = int(topn ** 0.5)
            for similar, _ in model.wv.most_similar(word, topn=topn):
                if similar not in seen:
                    q.put((-distance, similar))
                    seen.add(similar)

            time.sleep(1)
