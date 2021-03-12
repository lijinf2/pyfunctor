import sys
import os
if 'FACTMINE_HOME' in os.environ:
    pyfunctor_path = os.path.join(os.environ['FACTMINE_HOME'], 'third/pyfunctor')
    sys.path.append(pyfunctor_path)
else:
    sys.exit("please declara environment variable 'FACTMINE_HOME'")

import pdb

import csv_handler as csv_handler
import transform as transform

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline

def bart(sents, num_words = 100, margin = 20):
    bart = pipeline("summarization")
    article = transform.reduce_func(sents, lambda x, y: x + " " + y)

    min_words = num_words - margin
    max_words = num_words + margin

    result = bart(article, min_length = min_words, max_length = max_words) 
    return result[0]['summary_text']

def kmeans(sentences, topk, random_seed = 0):
    num_clusters = topk
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, random_state = random_seed)
    cluster_distance_pairs = model.fit_transform(X)   

    visited_idx = set() # incase num_clusters > actual #clusters
    result_sents = []
    for c in range(num_clusters):
        distances = transform.map_func(cluster_distance_pairs, lambda row : row[c])
        sort_dist_index = sorted(range(len(distances)), key = lambda idx : distances[idx])
        idx = 0
        while idx in visited_idx:
            idx += 1

        assert(idx < len(sort_dist_index))
        visited_idx.add(idx)

        idx = sort_dist_index[idx]
        result_sents.append(sentences[idx])

    return result_sents

class Vertex:
    def __init__(self, vid, nbs, init_pr):
        self.id = vid
        self.nbs = nbs
        self.pr = init_pr

    def get_pr(self):
        return self.pr

    def update(self, in_msgs):

        # gather 
        pr = 0.0
        for msg in in_msgs:
            pr += msg

        self.pr = 0.85 * pr + 0.15

        # scatter
        out_msgs = transform.map_func(range(len(self.nbs)), lambda nb : [nb, self.pr / len(self.nbs)])
        return out_msgs

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances

def textrank(sentences, topk, knn_graph = 5, iterations = 20):
    k_sents = topk
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    dist_matrix = euclidean_distances(X, X)

    # construct adj_list with top-k nbs

    # init vertices and message buffers
    vertices = []
    msg_buffer = []
    for vid in range(len(dist_matrix)):
        # build nbs
        dists = dist_matrix[vid]
        nbs = sorted(range(len(dists)), key = lambda i : dists[i])[:knn_graph]
        vertices.append(Vertex(vid, nbs, 0.15))
        msg_buffer.append([])

    def update_vertices():
        for vid in range(len(vertices)):
            msgs = vertices[vid].update(msg_buffer[vid])
            for nbid, pr in msgs:
                msg_buffer[nbid].append(pr)        

    for iter in range(iterations):
        update_vertices()

    node_idx = sorted(range(len(vertices)), key = lambda vid : -vertices[vid].get_pr())[:k_sents]

    result_sents = transform.map_func(node_idx, lambda idx : sentences[idx])

    return result_sents

def dedup_text(texts, method, output_num_text):
    result_texts = []
    if method == "textrank":
        result_texts = textrank(texts, output_num_text)

    elif method == "kmeans" :
        result_texts = kmeans(texts, output_num_text)

    else:
        print("method should be either kmeans or textrank")
        sys.exit()

    return result_texts

if __name__ == '__main__':
    input_path = sys.argv[1] 
    method = sys.argv[2]
    num_text = int(sys.argv[3])
    text_csv_idx = 1

    dataset = csv_handler.csv_readlines(input_path)
    texts = transform.map_func(dataset, lambda t : t[1])
    text_summarization = dedup_text(texts, method, num_text)
    transform.print_rows(text_summarization)
