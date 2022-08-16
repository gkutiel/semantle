from fire import Fire
from gensim.models import Word2Vec


def print_similar(word, n=20):
    model = Word2Vec.load('model.mdl')
    for w, _ in model.wv.most_similar(word, topn=n):
        print(w)


if __name__ == '__main__':
    Fire(print_similar)
