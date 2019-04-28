import json
from google.cloud import storage

bucketId = 'devspec'
bucket = None

def initiateBucketConnection(bucketId):
    client = storage.Client()
    bucket = client.get_bucket(bucketId)
    globals()['bucket'] = bucket
    return bucket

def getFilesFromBucket(fileName):
    blob = bucket.get_blob(fileName)
    json_data = blob.download_as_string()
    json_data = json_data.decode('utf8')
    return json.loads(json_data)

def storeFileinBucket(fileName,content):
    blob = bucket.blob(fileName)
    blob.upload_from_string(content)

def searchSubreddit(subname):
    prefix = '/'+str(subname)
    files = bucket.list_blobs(prefix=prefix,delimiter='/')
    counter = 0
    for f in files:
        counter +=1
    if counter > 0:
        return True
    else:
        return False