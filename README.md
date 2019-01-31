# Recommender Systems
This composed of two parts: **Recommendation-System** and **Recommendation/Prediction-with-PageRank**.


### Part 1
------
In this part of the project we have tried to improve the performance of a recommendation-system by using non-trivial
algorithms and also by performing parameters tuning.

#### Part 1-1 
The following tables shows the output from executing the "cross_validate" command on all algorithms made available by the [Surprise](http://surpriselib.com/) library. To use all CPU-cores on the machine we just have to set the parameter `n_jobs` on the cross_validate function to -1. The algorithms with better results than all basic algorithms are: **SVD**, **SVD++** and **KNNBaseline**. <br/>

**Normal Predictor** <br/>
 
| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|1.5028|1.5021|1.5066|1.5073|1.5042|1.5046|0.0020|
|Fit time|0.16|0.16|0.31|0.16|0.16|0.19|0.06|
|Test time|0.46|0.76|0.52|0.50|0.47|0.54|0.11|

**BaselineOnly** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|0.9176|0.9201|0.9174|0.9209|0.9225|0.9197|0.0020|
|Fit time|0.16|0.12|0.14|0.13|0.11|0.13|0.02|
|Test time|0.40|0.37|0.35|0.37|0.33|0.36|0.02|

**SVD** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|0.9068|0.9095|0.9050|0.9095|0.9101|0.9082|0.0020|
|Fit time|12.11|12.95|12.30|15.29|14.32|13.39|1.22
|Test time|0.53|0.49|0.49|0.52|0.53|0.51|0.02|
