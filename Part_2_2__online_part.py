
from sklearn.preprocessing import normalize
import networkx as nx
import numpy as np
import scipy
import json
import csv


def predict(pr_vectors, personalization_vector):
    pers_vector = {key:value/sum(personalization_vector.values()) for key, value in personalization_vector.items()}
    tmp_res = sum(pers_vector[category]*np.array(list(pr_vectors[category].values())) for category in pr_vectors)
    res = sorted([(value[0], tmp_res[index]) for index, value in enumerate(pr_vectors[list(pr_vectors.keys())[0]].items())], key=lambda x: x[1], reverse=True)
    for movie, score in res:
        print("{}, {}".format(movie, score))
    return res

if __name__ == "__main__":

    with open('pagerank_vectors.json') as f:
        pagerank_vectors = json.load(f)
    
    personalization_vector = {
        '1' : 1,
        '2' : 2,
        '3' : 3,
        '4' : 4,
        '5' : 5,
    }

    res = predict(pagerank_vectors, personalization_vector)