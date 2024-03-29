import retdata
import word2vec
import createbow
import trainTM
import classifier
import helpers
import gensim
import json
from collections import OrderedDict


isTrainedW2w = False
corexData = None
corexModel = None
numberOfTopics = 0
word2vecdata = None
w2w = None
subreddit = None
startTime = 0
endTime = 0
numberOfDiscussions = 2

'''Select the subreddit for analysis,
If a subreddit is already existing and has a W2W model trained then we do not need to retrain the W2W Model
'''
def w2wmodel(game):
    #if game == globals()['subreddit'] and globals()['isTrainedW2w'] == True:
        #return "True"
    #elif game == globals()['subreddit']:
        #w2wdata = retdata.retriveword2vecdata()
        #globals()['word2vecdata'] = w2wdata
        #globals()['w2w'] = word2vec.trainWord2Vec(word2vecdata,300)
        #globals()['isTrainedW2w'] = True
        #return "True"
    #else:
        #globals()['subreddit'] = game
        #For now the users input to this method is disregarded as we will only analyze feedback based on PUBATTLEGROUNDS
        #w2wdata = retdata.retriveword2vecdata()
        #globals()['word2vecdata'] = w2wdata
        #globals()['w2w'] = word2vec.trainWord2Vec(word2vecdata,300)
        #globals()['isTrainedW2w'] = True
        #return "True"
    model = gensim.models.KeyedVectors.load_word2vec_format('./modelworkplz.bin', binary=True,limit=100000)
    globals()['w2w']= model
    return "True"

default_anchors = '{"performance":["fps","ram","cpu","freeze","crash","gpu"],"gunplay":["gun","crosshair","shoot","recoil","control","spray"],"microtransactions":["crates","bp","skin","skins","camo"],"sounds":["footsteps","sound"],"maps":["erangel","map","maps","road","roads","compound"],"hackers":["anti","cheat","hackers","cheater","hacks"],"servers":["server","desync","lag","network","ping"]}'        
'''Input query parameters , End, Topics, Keywords
@start - Integer : Start time to consider discussions withing range
@end - Integer : End time to disregard discussions
@topics - Integer : Number of topics defined by the user
@keywords - String : Topic key word pairs as string

Returns: topic coherence score graph, keywords per each topic
'''
def analyze(start,end,topics=7,keywords=default_anchors,discussions=2):
    #print(keywords)
    keywords = str(keywords)
    keyword_dict=json.loads(keywords,object_pairs_hook=OrderedDict)
    #print(keyword_dict)
    keyword_list=[]
    topic_list=[]
    for topic,keywordset in keyword_dict.items():
        keyword_list.append(keywordset)
        topic_list.append(topic)
    #print(keyword_list)
    #print(topic_list)
    globals()['numberOfTopics'] = topics
    globals()['startTime'] = start
    globals()['endTime'] = end
    globals()['numberOfDiscussions'] = discussions
    #Note that this is not the start and endtime this is all the data we have scraped
    ldadata = retdata.retriveTMdata(1509494400,1539302400)
    globals()['corexData'] = ldadata
    bowmodel = createbow.createBBOW(ldadata['docs'])
    topic_model=trainTM.trainCorex(bowmodel['bow_mat'],bowmodel['bow_words'],ldadata['keys'],topics,keyword_list)
    globals()['corexModel'] = topic_model

    #return the topic cohession scores we will go with the default 20 top words example in this case.
    results = helpers.get_topic_cohission(topics,globals()['w2w'],topic_model)
    wordclouds = helpers.get_word_clouds(topics,keyword_list,10,topic_model)

    result_dict = dict()
    result_dict['scores'] = results
    result_dict['wordclouds'] = wordclouds

    return result_dict

def results():
    discussions_keys = classifier.uniqueFromDocTopicMatrix(globals()['numberOfTopics'],globals()['corexModel'].labels
    ,globals()['startTime'],globals()['endTime'],globals()['corexData']['keys'],globals()['corexData']['dicts'])
    #This will devicde discussions in each topic under 3 intention categires from ARDOC tool
    topic_subdiscussions = []
    i = 0
    for topic_keys in discussions_keys: 
        result=classifier.devideIntoCategories(discussions_keys[i],globals()['corexData']['dicts'])
        i = i+1
        topic_subdiscussions.append(result)

    title_selftext_vector_list = dict()
    chosen_discussion_list_keys = dict()
    i = 0
    for topic_group in topic_subdiscussions:
        topic_groups_vectors = dict()
        chosen_sub_discussion_list_keys = dict()
        for sub_topic, ids in topic_group.items():
            if len(ids)>0:
                subcategory_vector_list = []
                subcategory_key_list = []
                subcategory_vector_list=classifier.getTitleSelftextVectors(ids,globals()['corexData']['dicts'],globals()['w2w'])
                topic_groups_vectors[sub_topic]=subcategory_vector_list
                for item in subcategory_vector_list:
                    subcategory_key_list.append(item['key'])
                chosen_sub_discussion_list_keys[sub_topic] = subcategory_key_list
            
        title_selftext_vector_list[i]=topic_groups_vectors
        chosen_discussion_list_keys[i] = chosen_sub_discussion_list_keys
        i = i+1

    chosen_discussion_list = dict()
    i = 0
    for key,each_topic_item in title_selftext_vector_list.items():
        chosen_sub_discussion_list = dict()
        for sub_topic,sub_topic_items in each_topic_item.items():
            chosen_discussions = []
            chosen_discussions = classifier.getClusterSim(sub_topic_items,key,globals()['numberOfDiscussions'],globals()['corexModel'],globals()['w2w'])
            chosen_sub_discussion_list[sub_topic]=chosen_discussions
        chosen_discussion_list[key]=chosen_sub_discussion_list
    all_summaries = dict()
    for key,chosen_discussions_topic in chosen_discussion_list.items():
        single_topic_summary = dict()
        for sub_topic_name,listitems in chosen_discussions_topic.items():
            grouped_summaries=classifier.create_title_summaries(listitems,globals()['corexData']['dicts'],key,globals()['corexModel'],globals()['w2w'])
            final_summary = ""
            for discussion_summary in grouped_summaries:
                final_summary = final_summary + " " + discussion_summary['header'] + " " + discussion_summary['content']
            single_topic_summary[sub_topic_name] = final_summary
        all_summaries[key]=single_topic_summary

    return {"summaries":all_summaries,"Keys":chosen_discussion_list_keys}
