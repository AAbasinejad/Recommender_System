# Recommender Systems
This composed of two parts: **Recommendation-System** and **Recommendation/Prediction-with-PageRank**.


### Part 1
------
In this part of the project we have tried to improve the performance of a recommendation-system by using non-trivial
algorithms and also by performing parameters tuning.

#### Part 1.1 
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

#### Part 1.2
For this part we chose to run the SVD algorithm, because it has a good RMSE (but not the best) and a good
execution time, thus providing the best trade-off between execution time and RMSE result. We take in
consideration both fit time and test time, since fitting the algorithm only happens once, the test time is a more
important measure. <br/>

The Grid of Parameters used was the following: <br/>
```python
param_grid = {
'n_factors': [50, 100, 200],
'n_epochs': [10, 20, 30],
'lr_all': [0.002, 0.005, 0.01],
'reg_all': [0.01, 0.02, 0.04]
}
```
<br/>
The configuration of the best estimator was: <br/>
```python
{
'n_factors': 50,
'n_epochs': 30,
'lr_all': 0.005,
'reg_all': 0.04
}
```
<br/>
The average-RMSE of the best estimator was: **0.894** <br/>
The time taken to find the best estimator was **1582** seconds, or **26** minutes and **22** seconds. <br/>
A total of **8** CPU-cores were used. To use all the CPU-cores we set the parameter `n_jobs` on the GridSearchCV
method to -1.<br/>

### Part 2
------
In this part of the project we implement a recommendation/prediction method using a particular link-analysis procedure ([Topic Specific PageRank](https://nlp.stanford.edu/IR-book/html/htmledition/topic-specific-pagerank-1.html)). We used the R-Precision metric to evaulate the quality of the provided method.<br>

#### Part 2.1
To complete this part we filled the required pieces of code to complete the implementation of Topic Specific PageRank and got the following results: <br/>

|Average R-precision|Execution Time (seconds)|
|---|---|
|0.169513853715|244|
<br/>
The average R-Precision is averaged over all pairs (training-set, test-set), and the execution time refers to the time to execute all the functions (including loading/creating the graphs and so on...)<br/>

#### Part 2.2
In this part we had to implement a recommendation method based on Topic Specfic PageRank where the training part happens offline and the prediction happens online, meaning, PageRank is not computed on recommendation time. <br/>

**_Offline_**<br/>

The offline part is responsible for computing the PageRank vectors using the Topic Specific PageRank approach. The algorithm is as following:<br/>

1. Create the movie-movie graph where each node is a movie and if there is a relation between two movies they have a weigthed edge between them. Let's call this graph **_G_**.<br/>
2. Get the normalized adjacency matrix from **_G_**. This is the transition probability matrix of the Markov Chain, let's call it **_T_**.
3. For each movie category **_C_** create a vector where each element, <br/>

![](http://latex.codecogs.com/gif.latex?e_i%20%3D%20%5Cbegin%7Bcases%7D%20%26%20%5Cfrac%7B1%7D%7Bs%7D%20%5Ctext%7B%20if%20%7D%20i%5Cin%20C%20%5C%5C%20%26%20%5Ctext%7B0%20otherwise%7D%20%5Cend%7Bcases%7D)
<br/>

where **_S_** is the total number of movies in category **_C_**. This will be used to bias the teleportation on the transition probability matrix. Let's call this vector M<sub>i</sub> for category *i*. <br/>

4. Create initial PageRank vectors (one for each category) of size N (total number of nodes in the graph) and assign an initial value equal to ![](http://latex.codecogs.com/gif.latex?%5Cfrac1N) for each element in the vectors. This is the result of the PageRank calculation, let's call it Pr<sub>i</sub> for category *i*.  <br/>

5. Choose a value α between 0 and 1 to be one minus the teleportation probability (e.g. choose α = 0.8 to teleport with probability 0.2). <br/>

6. Run the following iteration for each Pr<sub>i</sub> until it converges to a steady state: <br/>

![](http://latex.codecogs.com/gif.latex?Pr_i%20%3D%20%5Calpha%20%5Ctimes%20Pr_i%20%5Ctimes%20T%20&plus;%20%281-%5Calpha%29%20%5Ctimes%20M_i)
<br/>

The result of this method is five PageRank vectors, one for each category. What this method does differently from normal PageRank is that during the iteration for category i the teleportation procedure only happens to movies in category i with uniform probability. This means that the random surfer when is teleported will not land in a movie outside category i, thus, creating a bias towards movies in category i. <br/>

**_Online_**<br/>

The online part is responsible for generating the sorted list of recommended movies based on a preference vector of the user, and is quite a simple procedure. <br/>
First we take the five PageRank vectors that were generated on the offline part, then we do a linear combination between these vectors times the normalized preference vector of the user. This will generate a list of PageRank values, one for each each movie in the original graph, that can be showed to the user as recommended movies. <br/>

