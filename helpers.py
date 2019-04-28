# Get Cosine Similarity of two vectors
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Get sentence vector by averaging all word vectors
def avg_sentence(sentence, wv):
  v = np.zeros(300)
  for w in sentence:
    if w in wv:
      v += wv[w]
  return v / len(sentence)

#get sentence vecors for each sentence
def generate_sent_vec(sentences = None):
    sent_vec_list = []
    for sentence in sentences:
        sentence = sentence.strip()
        sentvec=avg_sentence(sentence.split(), model.wv)
        sent_vec_list.append(sentvec)
        
    return sent_vec_list
              
#clean documents type 
# type 1 for removing punctuation including full stop
# type 2 to retain fullstops
def clean(text,tpe=0):
    punctuation = "!\"#$%&'()*+,-/:;<=>?@[\]^_`{|}~\n"
    if tpe == 1:
        punctuation +="."
    text=re.sub(r'http\S+', '', text)
    printable = set(string.printable)
    filter(lambda x: x in printable, text)
    regex1 = re.compile('[%s]' % re.escape(punctuation))
    text = regex1.sub('',text)
    text = text.replace('deleted', '')
    stripped_text = ''
    for c in text:
        stripped_text += c if len(c.encode(encoding='utf_8'))==1 else ''
    if tpe == 2:
        stripped_text=re.sub(r'(?<!\d)\.|\.(?!\d)', '*', stripped_text)
    return stripped_text

#For clustering 
def get_k_means_cluster(tfidf_matrix, num_clusters, is_list=True):
    from sklearn.cluster import KMeans

    #print ("Running k-means on " + str(num_clusters) + " clusters.")
    km = KMeans(n_clusters=num_clusters, verbose=0,random_state=3423,max_iter=100)

    km.fit(tfidf_matrix)
    # pickle km here.
    clusters = km.labels_
    #print(clusters)
    if is_list:
        clusters = km.labels_.tolist()
    return km, clusters

def determine_clusters(discs,length):
    range_val = 10
    if length <= range_val:
        range_val = length - 1
        
    vectors = discs
    topno =  2
    topscore = 0
    for num in range(2,range_val):
        km, cluster = get_k_means_cluster(vectors, num, is_list=False)
        curscore = run_silhoutte_analysis(vectors, cluster, num)
        if curscore >= topscore:
            topscore = curscore
            topno = num
    print("topno : " + str(topno) + " topscore : " + str(topscore))
    return topno

def run_silhoutte_analysis(X, cluster, num_cluster,plotgraph = False):

    sil_avg = silhouette_score(X, cluster)
    if plotgraph == True:

        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(10, 5)
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, 300 + (num_cluster + 1) * 10])
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        #print ("For number of clusters: " + str(num_cluster) + " average sil score:" + str(sil_avg))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster)
        y_lower = 10
        for i in range(num_cluster):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / num_cluster)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                            edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title("For number of clusters: " + str(num_cluster) + " average sil score:" + str(sil_avg))
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        # The vertical line for average silhoutte score of all the values
        ax1.axvline(x=sil_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        # # 2nd Plot showing the actual clusters formed
        plt.show()
    return sil_avg