###
### http://surpriselib.com/
###
import os
import pprint as p
import csv
from surprise import Dataset
from surprise import Reader

from surprise import NormalPredictor, BaselineOnly
from surprise import SVD, SVDpp, NMF
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import SlopeOne
from surprise import CoClustering

from surprise.model_selection import KFold
from surprise.model_selection import cross_validate


########################################
########################################
########################################


print

# path to dataset file
file_path = os.path.expanduser('./dataset/ratings.csv')

print ("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print ("Done.")
print


########################################
########################################
########################################


print 
print ("Performing splits...")
kf = KFold(n_splits=5, random_state=0) 
print ("Done.")
print 

print
current_algo = NormalPredictor()
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)
print
print ("NormalP Done!")


print
current_algo = BaselineOnly()
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)
print
print("Baseline Done!")



#Add your code here :)
current_algo = SVD()
print("SVD in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)


current_algo = SVDpp()
print("SVDpp in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)

current_algo = NMF()
print("NMF in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)

current_algo = KNNBasic()
print("KNN Basic in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)

current_algo = KNNBaseline()
print("KNNBaseline in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)

current_algo = KNNWithMeans()
print("KNNWithMeans in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)

current_algo = KNNWithZScore()
print("KNNWithZScore in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)

current_algo = SlopeOne()
print("SlopeOne in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)

current_algo = CoClustering()
print("CoClustering in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)