from corextopic import corextopic as ct

doc_topic_mat = None
anchors = None

def trainCorex(matrix,words,keys,topics,anchors):
    topic_model = ct.Corex(topics, seed=2)
    topic_model.fit(matrix, words=words, docs=keys, anchors=anchors, anchor_strength=8)
    topics = topic_model.get_topics()
    #globals()['topics'] = topics
    globals()['anchors'] = anchors
    globals()['doc_topic_mat'] = topic_model.labels
    print(topic_model.tc)
    return topic_model

def printTopWords():
    for n in range(globals()['topics']):
        topic_words,_ = zip(*topic_model.get_topics(topic=n))
        print('{}: '.format(n) + ','.join(topic_words))

def getDocTopicMatrix():
    return globals()['doc_topic_mat']