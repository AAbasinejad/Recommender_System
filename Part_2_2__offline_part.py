
from sklearn.preprocessing import normalize
import networkx as nx
import numpy as np
import scipy
import json
import csv

def create_graph(filename):
    g = nx.Graph()
    input_file = open(filename, 'r')
    input_file_csv_reader = csv.reader(input_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
    edges = [(int(line[0]), int(line[1]), int(line[2])) for line in input_file_csv_reader]
    g.add_weighted_edges_from(edges)
    return g

def get_movies_categories(filename):
    input_file = open(filename, 'r')
    input_file_csv_reader = csv.reader(input_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
    movies_categ = {i+1:list(map(int,line)) for i, line in enumerate(input_file_csv_reader)}
    return movies_categ

def pagerank(g, movies_categ, max_iter=100, alpha=0.85, tol=1.0e-6):
    number_of_nodes = len(g.nodes())
    adj_matrix = nx.adjacency_matrix(g)
    M = normalize(adj_matrix, norm='l1', axis=1)
    pr_vectors = {i:scipy.repeat(1/number_of_nodes, number_of_nodes) for i in movies_categ }
    ps = {category :np.array([0 if i+1 not in movies else 1/len(movies) for i in range(number_of_nodes)]) for category, movies in movies_categ.items()}
    for category in pr_vectors:
        for _ in range(max_iter):
            vector_last = pr_vectors[category].copy()
            pr_vectors[category] = alpha*pr_vectors[category]*M + (1-alpha)*ps[category]
            err = scipy.absolute(pr_vectors[category] - vector_last).sum()
            if err < number_of_nodes * tol:
                pr_vectors[category] = pr_vectors[category].tolist()
                break
    res = {key:{i+1:score for i, score in enumerate(value) } for key, value in pr_vectors.items()}
    with open('pagerank_vectors.json', 'w') as fp:
        json.dump(res, fp)
        print("File {} generated".format('pagerank_vectors.json'))
    return pr_vectors

if __name__ == "__main__":
    print("\nCreating graph...")
    graph = create_graph("dataset/movie_graph.txt")
    print("Graph created\n")
    movies_categ = get_movies_categories("dataset/category_movies.txt")
    print("Training topic specific PageRank...")
    pr = pagerank(graph, movies_categ)
    print("Done\n")