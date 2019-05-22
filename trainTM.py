from corextopic import corextopic as ct


default_anchors = [
            [ 'fps','ram','cpu','freeze','crash','gpu'],
            [ 'gun','crosshair','shoot','recoil','control','spray'], 
            ['crates','bp','skin','skins','camo'],
            ['footsteps','sound'],
            ['erangel','map','maps','road','roads','compound'],
            ['anti','cheat','hackers','cheater','hacks'],
            ['server','desync','lag','network','ping']
            ]
    
#topics = 7
doc_topic_mat = None
anchors = None

def trainCorex(matrix,words,keys,topics=7,anchors=default_anchors):
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