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
|Fit time(s)|0.16|0.16|0.31|0.16|0.16|0.19|0.06|
|Test time(s)|0.46|0.76|0.52|0.50|0.47|0.54|0.11|

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

**SVD++** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE(testset)|0.8945|0.893|0.8907|0.8938|0.8967|0.8937|0.002|
|Fit Time|1873.69|1888.47|1882.32|1877.38|764.35|1657.24|446.47|
|Test Time|29.85|25.59|28.03|29.91|11.55|24.98|6.9|

**NMF** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|0.9303|0.9382|0.9334|0.9392|0.9389|0.9360|0.0036|
|Fit time|14.22|17.60|15.25|15.50|17.49|16.01|1.32|
|Test time|0.69|0.41|0.43|0.42|0.49|0.49|0.11|

**KNNBasic** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|0.9468|0.9517|0.9505|0.9530|0.9560|0.9516|0.0030|
|Fit time|0.82|0.89|0.87|1.01|1.02|0.92|0.08|
|Test time|7.83|6.85|7.86|6.64|8.11|7.46|0.59|

**KNNBaseline** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|0.9051|0.9101|0.9065|0.9100|0.9107|0.9085|0.0022|
|Fit time|0.93|0.99|1.08|1.07|1.24|1.06|0.10|
|Test time|8.09|10.55|7.97|7.79|9.92|8.86|1.14|

**KNNWithMeans** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|0.9287|0.9347|0.9312|0.9331|0.9344|0.9324|0.0022|
|Fit time|1.27|1.38|1.04|0.84|1.05|1.12|0.19|
|Test time|7.10|9.48|7.31|7.18|8.51|7.92|0.93|

**KNNWithZScore** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|0.9272|0.9338|0.9307|0.9322|0.9330|0.9314|0.0023|
|Fit time|0.93|1.06|1.16|1.35|0.82|1.06|0.18|
|Test time|10.28|7.72|8.65|9.49|7.49|8.72|1.05|

**SlopeOne** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|0.9199|0.9253|0.9209|0.9236|0.9253|0.9230|0.0022|
|Fit time|4.23|3.42|3.31|3.30|4.48|3.75|0.50|
|Test time|13.53|8.99|9.13|8.91|11.91|10.49|1.89|

**CoClustering** <br/>

| |**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Fold 5**|**Mean**|**Std**|
|---|---|---|---|---|---|---|---|
|RMSE (testset)|0.9361|0.9473|0.9427|0.9405|0.9448|0.9423|0.0038|
|Fit time|2.71|2.65|2.41|2.94|2.35|2.61|0.22|
|Test time|0.72|0.42|0.38|0.38|0.37|0.45|0.14|

#### Part 1-2
