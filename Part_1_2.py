#Add your code here :)
import os
import time
from surprise import Dataset
from surprise import Reader

from surprise import SVD

from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV





print

# path to dataset file
file_path = os.path.expanduser('../1_1/dataset/ratings.csv')

print ("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print ("Done.")
print


print
print ("Performing splits...")
kf = KFold(n_splits=5, random_state=0)
print ("Done.")
print

current_algo = SVD()
print("SVD in progress!")
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)


param_grid = {
    'n_factors': [50, 100, 200],
    'n_epochs': [10, 20, 30], 
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.01, 0.02, 0.04],
}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5, n_jobs=-1)
start = time.time()
print("\nRunnning grid search...")
gs.fit(data)
end = time.time()
print("Done")
print("Time taken: {}\n".format(end-start))

# average RMSE score
print("\nBest RMSE")
print(gs.best_score['rmse'])
print("\nBest configuration")
# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])