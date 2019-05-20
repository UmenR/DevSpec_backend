import retdata
import word2vec
import createbow
import trainTM



word2vecdata = None
corexdata = None
corexbow = None
w2w = None
corex = None

#This method will get the data from storage and store it in variables
def getFeedback():
    print('1 ok')
    w2wdata = retdata.retriveword2vecdata()
    #Change later hardcode for now
    ldadata = retriveTMdata(1509494400,1539302400)

     globals()['word2vecdata'] = w2wdata
     globals()['corexdata'] = ldadata


def traininter():
    globals()['w2w'] = word2vec.trainWord2Vec(word2vecdata,300)
    print('W2W has been trained')
    globals()['corexbow'] = createbow.createBBOW(corexdata['docs'])
    print('BBOW has been trained')


def traincorex():
    trainTM.trainCorex(globals()['corexbow']['bow_mat'])



def analyze():
    getFeedback()
    print('2 ok')
    return "inanalyze"

