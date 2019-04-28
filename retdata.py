import os
import json

def retriveword2vecdata():
    sentences = []
    #Todo Replace with bucket storage 
    with open('pubg-dataset.txt', 'r') as f:
        for eachline in f:
            #clean the /n
            eachline = eachline.strip('\n')
            #break each line to sentences
            eachline = eachline.split('.')
            for eachsentence in eachline:
                if eachsentence != '' and eachsentence.isspace() == False:
                    eachsentence = eachsentence.split()
                    sentences.append(eachsentence)
    return sentences


def retriveTMdata(start,end):
    oneDay = 86400
    startTime = start + 1
    data = []
    plainText = []
    keys = []
    classifications = []
    dicts=[]
    count = 0
    while(startTime < end):
        try:
            #Todo Replace with bucket storage
            script_dir = os.path.dirname('__file__')
            rel_path = "Dataset/"+str(startTime)+".json"
            abs_file_path = os.path.join(script_dir, rel_path)
            with open(abs_file_path) as f:
                data = json.load(f)
            
            for document in data:
                if "im a bot" in document['gencomments']:
                    count+=1
                    continue
                else:
                    concatString = document['title'] +  ' . ' + document['selftext'] + ' . ' + document['gencomments']
                    uncleaned = concatString
                    concatString = clean(concatString)
                    plainText.append(str.lower(concatString))
                    keys.append(document['id'])
                    classifications.append(document['classifierresults'])
                    dicts.append({'key':document['id'],'content':str.lower(concatString),'class':document['classifierresults'],
                                 'title':str.lower(document['title']),'selftext':str.lower(document['selftext']),'comments':str.lower(document['gencomments']),
                                 'uncleaned':str.lower(uncleaned),'time':startTime})
            startTime += oneDay
        except:
            startTime += oneDay
            
    return {'keys':keys,'docs':plainText,'count':count,'classify':classifications,'dicts':dicts}