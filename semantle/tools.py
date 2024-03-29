import re

from gensim.models import Word2Vec
from tqdm import tqdm


def seed_txt():
    seed = [
        'אדם',
        'אהבה',
        'בית',
        'משפחה',
        'עבודה',
        'חברים',
        'תרבות',
        'פוליטיקה',
        'חינוך',
        'כלכלה',
        'בריאות']

    word_re = re.compile(r'[אבגדהוזחטיכלמנסעפצקרשתךםןףץ]+')

    model = Word2Vec.load('model.mdl')
    wv = model.wv
    for _ in tqdm(range(100)):
        res = wv.most_similar(
            negative=seed,
            topn=1)
        seed.extend([w for w, _ in res])  # type: ignore

    seed = [w for w in seed if word_re.fullmatch(w)]
    with open('seed.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(seed))


if __name__ == '__main__':
    model = Word2Vec.load('model.mdl')
    wv = model.wv
    words = wv.most_similar('בודד', topn=1000)
    words = [w for w, _ in words if len(w) == 4 and 'ח' in w]
    with open('tmp.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(words))
