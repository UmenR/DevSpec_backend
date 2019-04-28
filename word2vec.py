from gensim.models import Word2Vec

def trainWord2Vec(data,dim):
    model = Word2Vec(data,size=dim)
    return model