

def get_comment_summary(comments,title,topic_vector):          
    comments = clean(comments,2)
    title= clean(title,1)
    title_vector = avg_sentence(title.strip().split(),model.wv)
    sentences = comments.split('*')
    filteredsentences = []
    filteredsentences = [element for element in sentences if len(element) > 0 and element.isspace()==False and element!='']
    filteredsentences = [element for element in filteredsentences if len(element) > 10]
    scoredsentences = []
    summary =''
    for sentence in filteredsentences:
        vector = avg_sentence(sentence.strip().split(),model.wv)
        titlesim = cosine_sim(title_vector,vector)
        scoredsentences.append({'text':sentence,'score':titlesim})
        
    scoredsentences.sort(key=lambda item:item['score'],reverse=True)
    scoredsentences = scoredsentences[:5]
    for sentence in scoredsentences:
        #print(sentence)
        summary += sentence['text'] + '.'
        
    return summary

def create_title_summaries(selected,dicts,topic):
    topic_sent = ""
    topic_prob_tuple = topic_model.get_topics(topic=topic, n_words=25)
    for word,prob in topic_prob_tuple:
        if prob > 0:
            topic_sent +=  word + " "
    
    topic_sent=topic_sent.strip()
    topic_sent_vec = avg_sentence(topic_sent.split(),model.wv)
    summary = []
    clusters =[]
    for selection in selected:
        for item in dicts:
            if selection['key'] == item['key']:
                #print('--'*40)
                concatstring = item['selftext']
                concatstring = clean(concatstring,tpe=2)
                sentences = concatstring.split('*')
                filteredsentences = []
                filteredsentences = [element for element in sentences if len(element) > 0 and element.isspace()==False and 
                                     element!='']
                #filteredsentences = [element for element in filteredsentences if len(element) > 20]
                scoredvectors = []
                scoredsentences = []
                for sentence in filteredsentences:
                    vector = avg_sentence(sentence.strip().split(),model.wv)
                    scoredsentences.append({'sentence':sentence,'vec':vector})
                    scoredvectors.append(vector)
                    
                n_clusters = determine_clusters(scoredvectors,len(scoredvectors))
                kmeans = KMeans(n_clusters=n_clusters)
                if  len(scoredvectors) >= n_clusters:
                    
                    kmeans.fit(scoredvectors)
                    for i in range(n_clusters):
                        clusters.append(np.where(i == kmeans.labels_)[0])
                        
                    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, scoredvectors)
                    indexes = []
                    for i in range(n_clusters):
                        indexes.append(closest[i])
                    
                    titlesum = ""
                    indexes.sort()
                    for idx in indexes:
                        titlesum = titlesum +"."+ scoredsentences[idx]['sentence']
                    titlesentences = clean(item['title'],tpe=2).split('*')
                    fintitlesent = ''
                    fintitlescore = -3
                    
                    if len(titlesentences) > 1:
                        for titlesentence in titlesentences:
                            if titlesentence != '' and titlesentence.isspace() == False: 
                                vector = avg_sentence(titlesentence.strip().split(),model.wv)
                                score = cosine_sim(topic_sent_vec,vector)
                                
                                if score > fintitlescore:
                                    fintitlescore = score
                                    fintitlesent = titlesentence            
                    else:
                        fintitlesent = titlesentences[0]
                        
                    
                    titlesum = fintitlesent + '.' + titlesum
                    summarizedcomments = get_comment_summary(comments=item['comments'],title=item['title']+'.'+
                                                             item['selftext'],topic_vector=topic_sent_vec)
                    summary.append({'header':titlesum,'content':summarizedcomments,'key':item['key']})
                    
                else:
                    titlesum = ""
                    titlesentences = clean(item['title'],tpe=2).split('*')
                    fintitlesent = ''
                    fintitlescore = -3
                    
                    if len(titlesentences) > 1:
                        for titlesentence in titlesentences:
                            if titlesentence != '' and titlesentence.isspace() == False: 
                                vector = avg_sentence(titlesentence.strip().split(),model.wv)
                                score = cosine_sim(topic_sent_vec,vector)          
                                if score > fintitlescore:
                                    fintitlescore = score
                                    fintitlesent = titlesentence            
                    else:
                        fintitlesent = titlesentences[0]
                    
                    titlesum = fintitlesent + '.' + titlesum
                    for sent in scoredsentences:
                        titlesum = titlesum + " . " + sent['sentence']
                    
                    summarizedcomments = get_comment_summary(comments=item['comments'],title=item['title']+'.'+item['selftext'],topic_vector=topic_sent_vec)
                    summary.append({'header':titlesum,'content':summarizedcomments,'key':item['key']})
                    
    return summary    