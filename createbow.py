import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ss



def createBBOW(docs):
    vectorizer = CountVectorizer(stop_words='english', max_features=20000, binary=True)
    doc_word = vectorizer.fit_transform(docs)
    doc_word = ss.csr_matrix(doc_word)
    doc_word.shape
    words = list(np.asarray(vectorizer.get_feature_names()))
    not_digit_inds = [ind for ind,word in enumerate(words) if not word.isdigit()]
    doc_word = doc_word[:,not_digit_inds]
    words = [word for ind,word in enumerate(words) if not word.isdigit()]
    not_short_inds = [ind for ind,word in enumerate(words) if not len(word)<2]
    doc_word = doc_word[:,not_short_inds]
    words = [word for ind,word in enumerate(words) if not len(word)<2]
    not_long_inds = [ind for ind,word in enumerate(words) if not len(word)>14]
    doc_word = doc_word[:,not_long_inds]
    words = [word for ind,word in enumerate(words) if not len(word)>14]
    doc_word.shape

    return {"bow_words":words,"bow_mat":doc_word}