from gensim.models import Word2Vec

def trainWord2Vec(data,dim=300):
    model = Word2Vec(data,size=dim)
    return model