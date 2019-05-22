from __future__ import print_function
import json
import os
import requests
import gensim
import re
import string
import scipy.sparse as ss
import pandas as pd
import math
import rouge
import nltk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import helpers
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from itertools import repeat
from pprint import pprint
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.cloud import storage
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from pythonrouge.pythonrouge import Pythonrouge
from math import ceil
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from word_cloud.word_cloud_generator import WordCloud
import warnings
warnings.filterwarnings('ignore')

# 1. provide all info for this method (all keys and all dicts are from the retdata.py files return value)
def uniqueFromDocTopicMatrix(num_topics,doc_topic_matrix,start_time,end_time,all_keys,all_dicts):
    documents_per_topic = []
    for ind in range(num_topics):
        documents_per_topic.append([])
    
    key_index = 0
    for row in doc_topic_matrix:
        is_one_topic = False
        topic_count = 0
        for topic in row:
            if topic == True:
                topic_count+=1
        
        #****CHANGE CONDITION TO INCLUDE DISCUSSIONS THAT APPEAR IN MULTIPLE TOPICS*****
        if topic_count == 1:
            for index,topic in np.ndenumerate(row):
                if topic == True and isInTimeRange(all_keys[key_index],all_dicts,start_time,end_time):
                    documents_per_topic[index[0]].append(all_keys[key_index])
           
        key_index+=1
        
    return documents_per_topic

#topicwordmatrix = unique_from_doc_topic(8,topic_model.labels,1509494400,1539302400)

def isInTimeRange(key,dicts,start_time,end_time):
    for item in dicts:
        if item['key'] == key:
            if item['time'] >= start_time and item['time']<=end_time:
                return True
            else:
                return False

#This will categorize every discussion regardless of the topic under the 5 subcategories
# 2. filtered_keys = output from uniqueFromDocTopicMatrix dictionaries = retdata.py
def devideIntoCategories(filterd_keys,dictionaries):
    other = []
    informative= []
    bug = []
    for filterd_key in filterd_keys:
        for dictionary in dictionaries:         
            if filterd_key == dictionary['key']:
                if int(dictionary['class']['other'])>0:
                    other.append(dictionary['key'])
                if int(dictionary['class']['bug'])>0:
                    bug.append(dictionary['key'])
                if int(dictionary['class']['inforeq'])>0 or int(dictionary['class']['infogive'])>0 or int(dictionary['class']['suggestion'])>0:
                    informative.append(dictionary['key'])
                    
    return {'ohter':other,'informative':informative,'bug':bug}

# 3. the out put from No 2 , is a 5 item dict pass any dict item to the following as ids
#dictionaries = retdata.py
def getTitleSelftextVectors(ids,dictionaries,w2vModel):
    id_vector_map =[]
    for ide in ids:
        for dic in dictionaries:
            if ide == dic['key']:
                titleandselftext = dic['title'] + ' . ' + dic['selftext']
                titleandselftext = re.sub(r'&\S+','',titleandselftext)
                titleandselftext = helpers.clean(titleandselftext,1)
                
                vector=helpers.avg_sentence(titleandselftext.split(),w2vModel.wv)
                if np.any(vector) == True:
                    id_vector_map.append({'key':dic['key'],'vec':vector,'titself':titleandselftext})
    return id_vector_map

# helper for method below
def removeBelowMedian(subdiscussions,topic,topic_model,w2vModel):
    topic_sent = ""
    topic_prob_tuple = topic_model.get_topics(topic=topic, n_words=25)
    for word,prob in topic_prob_tuple:
        if prob > 0:
            topic_sent +=  word + " "
    
    topic_sent=topic_sent.strip()
    topic_sent_vec = helpers.avg_sentence(topic_sent.split(),w2vModel.wv)
    
    scored_items = []
    for item in subdiscussions:
        sim = helpers.cosine_sim(topic_sent_vec,item['vec'])
        scored_items.append({'key':item['key'],'vec':item['vec'],'sim':sim,'titself':item['titself']})
    
    scored_items.sort(key=lambda item:item['sim'], reverse=False)
    is_even = False
    if len(scored_items) % 2 == 0:
        is_even = True
    
    median = 0
    if is_even == True:
        bottom = len(scored_items)/2
        top = bottom + 1
        median = (scored_items[int(bottom)]['sim'] + scored_items[int(top)]['sim'])/2
    else:
        middle = (len(scored_items) + 1)/2
        median = scored_items[middle]['sim']
    
    filtered_items = []
    for item in scored_items:
        if item['sim'] >= median:
            filtered_items.append(item)
            
    filtered_items.sort(key=lambda item:item['sim'], reverse=True)
    
    return filtered_items

#4 The following method will get discussions for a certain topic under a sub category such as bug etc.
# subdiscussions = 3 , topic = any topic from the topic list , sentnum = user input
#most simillar after clustering approach
def getClusterSim(subdiscussions,topic,sentnum):
    if len(subdiscussions) / 2  >= sentnum * 2:
        median_removed_list = remove_below_median(subdiscussions,topic)
        if len(median_removed_list) > sentnum:
            clusterlist = get_clusters(median_removed_list)
            finallist = choose_discussions(clusterlist,sentnum)
            return finallist
        else:
            median_removed_list.sort(key=lambda item:item['sim'], reverse=True)
            return median_removed_list
    elif len(subdiscussions) == sentnum:
        return subdiscussions
    else:
        clusterlist = get_clusters(subdiscussions)
        finallist = choose_discussions(clusterlist,sentnum)
        return finallist
        
#chosendiscussions = get_cluster_sim(testing,0,20)

# helper for above
def getClusters(discussions):
    vectorlist = []
    ids = []
    stringlist = []
    clusters =[]
    for vector in discussions:
        vectorlist.append(vector['vec'])
        ids.append(vector['key'])
        stringlist.append(vector['titself'])
            
    n_clusters = helpers.determineClusters(vectorlist, len(vectorlist))
    kmeans = KMeans(n_clusters=n_clusters)
    
    kmeans = kmeans.fit(vectorlist)
    for i in range(n_clusters):
        clusters.append(np.where(i == kmeans.labels_)[0])
    clusterlist = []
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectorlist)
            #Get the most simillar to the cluster center and score according to similarrity
    for i in range(n_clusters):
        currentcluster = clusters[i]
        appendcluster = []
        clstidx = closest[i]
        clstelement = vectorlist[clstidx]
        for element in currentcluster:
            sim = helpers.cosine_sim(clstelement,vectorlist[element])
            if math.isnan(sim)== False:
                appendcluster.append({'key':ids[int(element)],'sim':sim,'txt':stringlist[int(element)],
                                    'vec':vectorlist[int(element)]})
        appendcluster.sort(key=lambda item:item['sim'], reverse=True)
        clusterlist.append(appendcluster)
        return clusterlist

def chooseDiscussions(clusters,num):
    finaldisc = []
    filterdclusters = clusters
    if len(filterdclusters) == num:
        for cluster in filterdclusters:
            finaldisc.append(cluster[0])
            
    elif len(filterdclusters) > num:
        alldiscussions = []
        for cluster in filterdclusters:
            alldiscussions.append(cluster[0])
            
        alldiscussions.sort(key=lambda item:item['sim'], reverse=True)
        finaldisc = alldiscussions[:num]
        
    else:
        allitems = 0
        for cluster in filterdclusters:
            allitems+=len(cluster)
        #print('all items ' + str(allitems))
        if allitems <= num:
            for cluster in filterdclusters:
                for element in cluster:
                    finaldisc.append(element)
        
        else:
            items_per_cluster = int(num/len(filterdclusters))
            lowest_number = items_per_cluster
            for cluster in filterdclusters:
                if len(cluster) < lowest_number:
                    lowest_number = len(cluster)
            
            if lowest_number == items_per_cluster:
                #print(lowest_number)
                for cluster in filterdclusters:
                    items_to_append = cluster[:lowest_number]
                    for item in items_to_append:
                        finaldisc.append(item)
            else:
                remaining_discussions = []
                for cluster in filterdclusters:
                    finaldisc.append(cluster[:lowest_number])
                    if len(cluster)>lowest_number:
                        items_to_append = cluster[:lowest_number]
                        for item in items_to_append:
                            remaining_discussions.append(item)
                            
                remaining_item_count = num - len(finaldisc)
                remaining_discussions.sort(key=lambda item:item['sim'], reverse=True)
                items_to_append = remaining_discussions[:remaining_item_count]
                for item in items_to_append:
                    finaldisc.append(item)    
    #print('final len : '+ str(len(finaldisc)))
    return finaldisc


def get_comment_summary(comments,title,topic_vector,w2wmodel):          
    comments = clean(comments,2)
    title= clean(title,1)
    title_vector = helpers.avg_sentence(title.strip().split(),w2wmodel.wv)
    sentences = comments.split('*')
    filteredsentences = []
    filteredsentences = [element for element in sentences if len(element) > 0 and element.isspace()==False and element!='']
    filteredsentences = [element for element in filteredsentences if len(element) > 10]
    scoredsentences = []
    summary =''
    for sentence in filteredsentences:
        vector = helpers.avg_sentence(sentence.strip().split(),model.wv)
        titlesim = helpers.cosine_sim(title_vector,vector)
        scoredsentences.append({'text':sentence,'score':titlesim})
        
    scoredsentences.sort(key=lambda item:item['score'],reverse=True)
    scoredsentences = scoredsentences[:5]
    for sentence in scoredsentences:
        #print(sentence)
        summary += sentence['text'] + '.'
        
    return summary

def create_title_summaries(selected,dicts,topic,topic_model,w2wmodel):
    topic_sent = ""
    topic_prob_tuple = topic_model.get_topics(topic=topic, n_words=25)
    for word,prob in topic_prob_tuple:
        if prob > 0:
            topic_sent +=  word + " "
    
    topic_sent=topic_sent.strip()
    topic_sent_vec = helpers.avg_sentence(topic_sent.split(),w2wmodel.wv)
    summary = []
    clusters =[]
    for selection in selected:
        for item in dicts:
            if selection['key'] == item['key']:
                #print('--'*40)
                concatstring = item['selftext']
                concatstring = helpers.clean(concatstring,tpe=2)
                sentences = concatstring.split('*')
                filteredsentences = []
                filteredsentences = [element for element in sentences if len(element) > 0 and element.isspace()==False and 
                                     element!='']
                #filteredsentences = [element for element in filteredsentences if len(element) > 20]
                scoredvectors = []
                scoredsentences = []
                for sentence in filteredsentences:
                    vector = helpers.avg_sentence(sentence.strip().split(),w2wmodel.wv)
                    scoredsentences.append({'sentence':sentence,'vec':vector})
                    scoredvectors.append(vector)
                    
                n_clusters = helpers.determineClusters(scoredvectors,len(scoredvectors))
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
                    titlesentences = helpers.clean(item['title'],tpe=2).split('*')
                    fintitlesent = ''
                    fintitlescore = -3
                    
                    if len(titlesentences) > 1:
                        for titlesentence in titlesentences:
                            if titlesentence != '' and titlesentence.isspace() == False: 
                                vector = helpers.avg_sentence(titlesentence.strip().split(),w2wmodel.wv)
                                score = helpers.cosine_sim(topic_sent_vec,vector)
                                
                                if score > fintitlescore:
                                    fintitlescore = score
                                    fintitlesent = titlesentence            
                    else:
                        fintitlesent = titlesentences[0]
                        
                    
                    titlesum = fintitlesent + '.' + titlesum
                    summarizedcomments = get_comment_summary(comments=item['comments'],title=item['title']+'.'+
                                                             item['selftext'],topic_vector=topic_sent_vec,w2wmodel=w2wmodel)
                    summary.append({'header':titlesum,'content':summarizedcomments,'key':item['key']})
                    
                else:
                    titlesum = ""
                    titlesentences = helpers.clean(item['title'],tpe=2).split('*')
                    fintitlesent = ''
                    fintitlescore = -3
                    
                    if len(titlesentences) > 1:
                        for titlesentence in titlesentences:
                            if titlesentence != '' and titlesentence.isspace() == False: 
                                vector = helpers.avg_sentence(titlesentence.strip().split(),w2wmodel.wv)
                                score = helpers.cosine_sim(topic_sent_vec,vector)          
                                if score > fintitlescore:
                                    fintitlescore = score
                                    fintitlesent = titlesentence            
                    else:
                        fintitlesent = titlesentences[0]
                    
                    titlesum = fintitlesent + '.' + titlesum
                    for sent in scoredsentences:
                        titlesum = titlesum + " . " + sent['sentence']
                    
                    summarizedcomments = get_comment_summary(comments=item['comments'],title=item['title']+'.'+item['selftext'],topic_vector=topic_sent_vec,w2wmodel=w2wmodel)
                    summary.append({'header':titlesum,'content':summarizedcomments,'key':item['key']})
                    
    return summary