import csv
import pprint as pp
import networkx as nx
import itertools as it
import math
import scipy.sparse
import random



def pagerank(M, N, nodelist, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, dangling=None):
	if N == 0:
		return {}
	S = scipy.array(M.sum(axis=1)).flatten()
	S[S != 0] = 1.0 / S[S != 0]
	Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
	M = Q * M
	
	# initial vector
	x = scipy.repeat(1.0 / N, N)
	
	# Personalization vector
	if personalization is None:
		p = scipy.repeat(1.0 / N, N)
	else:
		missing = set(nodelist) - set(personalization)
		if missing:
			#raise NetworkXError('Personalization vector dictionary must have a value for every node. Missing nodes %s' % missing)
			print
			print 'Error: personalization vector dictionary must have a value for every node'
			print
			exit(-1)
		p = scipy.array([personalization[n] for n in nodelist], dtype=float)
		#p = p / p.sum()
		sum_of_all_components = p.sum()
		if sum_of_all_components > 1.001 or sum_of_all_components < 0.999:
			print
			print "Error: the personalization vector does not represent a probability distribution :("
			print
			exit(-1)
	
	# Dangling nodes
	if dangling is None:
		dangling_weights = p
	else:
		missing = set(nodelist) - set(dangling)
		if missing:
			#raise NetworkXError('Dangling node dictionary must have a value for every node. Missing nodes %s' % missing)
			print
			print 'Error: dangling node dictionary must have a value for every node.'
			print
			exit(-1)
		# Convert the dangling dictionary into an array in nodelist order
		dangling_weights = scipy.array([dangling[n] for n in nodelist], dtype=float)
		dangling_weights /= dangling_weights.sum()
	is_dangling = scipy.where(S == 0)[0]

	# power iteration: make up to max_iter iterations
	for _ in range(max_iter):
		xlast = x
		x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
		# check convergence, l1 norm
		err = scipy.absolute(x - xlast).sum()
		if err < N * tol:
			return dict(zip(nodelist, map(float, x)))
	#raise NetworkXError('power iteration failed to converge in %d iterations.' % max_iter)
	print
	print 'Error: power iteration failed to converge in '+str(max_iter)+' iterations.'
	print
	exit(-1)


def create_graph_set_of_users_set_of_items(user_item_ranking_file):
	graph_users_items = {}
	all_users_id = set()
	all_items_id = set()
	#g = nx.DiGraph()
	g = nx.Graph()
	input_file = open(user_item_ranking_file, 'r')
	input_file_csv_reader = csv.reader(input_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
	for line in input_file_csv_reader:
		user_id = int(line[0])
		item_id = int(line[1])
		g.add_edge(user_id, item_id)
		all_users_id.add(user_id)
		all_items_id.add(item_id)
	input_file.close()
	graph_users_items['graph'] = g
	graph_users_items['users'] = all_users_id
	graph_users_items['items'] = all_items_id
	return graph_users_items


def create_item_item_graph(graph_users_items):
	g = nx.bipartite.weighted_projected_graph(graph_users_items['graph'], graph_users_items['items'])
	return g

def create_preference_vector_for_teleporting(user_id, graph_users_items):
	preference_vector = {}
	# Your code here ;)
	user_neighbors = graph_users_items['graph'][user_id]
	prob = 1/float(len(user_neighbors))
	for item in graph_users_items['items']:
		try:
			user_neighbors[item]
			preference_vector[item] = prob
		except KeyError:
			preference_vector[item] = 0
	return preference_vector
	

def create_ranked_list_of_recommended_items(page_rank_vector_of_items, user_id, training_graph_users_items):
	# This is a list of 'item_id' sorted in descending order of score.
	sorted_list_of_recommended_items = []
	# You can obtain this list from a list of [item, score] pairs sorted in descending order of score.

	# Your code here ;)
	user_neighbors = list(training_graph_users_items['graph'].neighbors(user_id))
	for item in user_neighbors:
		try:
			del page_rank_vector_of_items[item]
		except KeyError:
			pass
	sorted_list_of_recommended_items =  [recom[0] for recom in sorted(page_rank_vector_of_items.items(), key = lambda x: x[1], reverse=True)]
	return sorted_list_of_recommended_items




def r_precision(user_id, sorted_list_of_recommended_items, test_graph_users_items):
	R_Precision = 0.
	# Your code here ;)
	ground_truth = list(test_graph_users_items['graph'].neighbors(user_id))
	k = len(ground_truth)
	predicted = set(sorted_list_of_recommended_items[:k])
	inter = predicted.intersection(set(ground_truth))
	R_Precision = len(inter)/float(k)
	return R_Precision

