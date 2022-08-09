import os
import heapq as hq
import requests
import time
import json
from tqdm import tqdm
from gensim.models import Word2Vec


def get(word):
    url = "https://semantle-he.herokuapp.com/api/distance?word=" + word
    r = requests.get(url).json()
    assert r is not None
    r['word'] = word
    return r


if __name__ == '__main__':
    model = Word2Vec.load('model.mdl')
    seen = set()
    q = []

    def init_q(words):
        for word in words:
            hq.heappush(q, (-950, word))

    init_q([
        'אוכל', 'אמבטיה', 'אנגלית', 'אמיץ',
        'בגד', 'בובה',
        'גאוותן', 'גג',
        'חביתה', 'חגורה', 'חדר', 'חומר', 'חפץ', 'חזק', 'חצאית',
        'טיגון',
        'ילד', 'ילדה',
        'כורסה',
        'מחבת', 'מחשב', 'מיטה', 'מטוס', 'מכשיר', 'מנוע', 'מקרר', 'משקל',
        'נער', 'נערה', 'נוזל',
        'ספר',
        'עיניים', 'עצמי', 'עקשן',
        'פנים', 'פעולה',
        'קציר', 'קצפת',
        'רגש', 'רהיט', 'רצפה',
        'שולחן', 'שעון', 'שיער',
        'תחושה', 'תפקיד'])

    best_distance = -1
    best_word = None
    bar = tqdm()
    with open('words.json', 'w', encoding='utf-8') as f:
        while q:
            p, word = hq.heappop(q)

            if word in seen:
                continue

            seen.add(word)
            r = get(word)

            if not r['similarity']:
                continue

            print(json.dumps(r), file=f)

            distance = r['distance']

            if distance == 1_000:
                print('\n' * 3)
                print('*' * 20)
                print('found', word)
                print('*' * 20)
                break

            if distance > best_distance:
                best_distance = distance
                best_word = word
                os.system(f'''
                    osascript -e 'display notification "{best_distance}" with title "{best_word}"'
                    ''')

            bar.update()
            bar.set_description(f'{best_word} {best_distance}')

            topn = int(max(1, distance))
            topn = topn ** 1.6
            topn = int(topn / 630)
            topn = max(1, topn)

            for similar, _ in model.wv.most_similar(word, topn=topn):
                hq.heappush(q, (-distance, similar))

            time.sleep(1)
