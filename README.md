# WeblogChallenge
This is an interview challenge for Paytm Labs. Please feel free to fork. Pull Requests will be ignored.

The challenge is to make make analytical observations about the data using the distributed tools below.

## Processing & Analytical goals:

1. Sessionize the web log by IP. Sessionize = aggregrate all page hits by visitor/IP during a fixed time window.
    https://en.wikipedia.org/wiki/Session_(web_analytics)
	
```
run BasicProcess.java
```

2. Determine the average session time

```
+-----------------+
|      avg(elapse)|
+-----------------+
|89.55637293816706|
+-----------------+
```

3. Determine unique URL visits per session. To clarify, count a hit to a unique URL only once per session.

```
+---------------+----------------+
|             ip|unique_url_count|
+---------------+----------------+
| 182.18.177.110|               1|
|117.215.137.201|               6|
|180.151.201.218|               6|
|   59.177.234.8|               2|
|  182.66.45.218|              57|
|     1.38.20.34|              14|
| 122.177.242.66|               1|
| 202.131.98.131|               3|
|  103.50.83.146|               3|
| 115.250.95.174|              10|
|   59.99.156.81|               1|
|122.160.211.198|               6|
|117.223.209.111|              13|
|   112.79.37.91|               2|
|203.129.218.178|               2|
|   115.97.26.99|               1|
|  117.240.56.66|               8|
|117.213.216.174|               1|
| 168.235.194.72|               2|
|   113.193.33.7|               1|
+---------------+----------------+
```

4. Find the most engaged users, ie the IPs with the longest session times

```
+---------------+--------------+
|             ip|session_length|
+---------------+--------------+
|  220.226.206.7|        5894.0|
|   52.74.219.71|        5256.0|
|  119.81.61.166|        5251.0|
|  54.251.151.39|        5231.0|
| 121.58.175.128|        4946.0|
|  106.186.23.95|        4929.0|
|   125.19.44.66|        4650.0|
|  54.169.191.85|        4611.0|
| 180.179.213.94|        4387.0|
| 54.252.254.204|        4271.0|
| 122.252.231.14|        4168.0|
| 180.179.213.71|        4143.0|
|   207.46.13.22|        4050.0|
| 176.34.159.236|        3928.0|
| 54.255.254.236|        3886.0|
|168.235.197.212|        3822.0|
|   54.232.40.76|        3802.0|
|  54.243.31.236|        3774.0|
|  116.50.59.180|        3723.0|
| 177.71.207.172|        3704.0|
+---------------+--------------+
```

## Additional questions for Machine Learning Engineer (MLE) candidates:
1. Predict the expected load (requests/second) in the next minute

- Use python + sklearn + numpy + pandas + scipy
```
GBM: Root Mean Squared Error (RMSE) on test data = 67.7

Randomforest: Root Mean Squared Error (RMSE) on test data =70.35

SVR: Root Mean Squared Error (RMSE) on test data = 68.4
```

- Use Java + Spark ML

```
GBTRegressor: Root Mean Squared Error (RMSE) on test data = 67.05
NumTree:200	MaxIter:200	MaxDepth:3

RandomforestRegressor: Root Mean Squared Error (RMSE) on test data = 85.72

LinearRegression: Root Mean Squared Error (RMSE) on test data = 67.25
Elastic:0.1	Reg:0.01
```

2. Predict the session length for a given IP

- Use python + scipy + sklearn

3. Predict the number of unique URL visits by a given IP

- Use python + scipy + sklearn


### Tools allowed (in no particular order):
- Spark (any language, but prefer Scala or Java)
- Pig
- MapReduce (Hadoop 2.x only)
- Flink
- Cascading, Cascalog, or Scalding


### Additional notes:
- You are allowed to use whatever libraries/parsers/solutions you can find provided you can explain the functions you are implementing in detail.
- IP addresses do not guarantee distinct users, but this is the limitation of the data. As a bonus, consider what additional data would help make better analytical conclusions
- For this dataset, complete the sessionization by time window rather than navigation. Feel free to determine the best session window time on your own, or start with 15 minutes.
- The log file was taken from an AWS Elastic Load Balancer:
http://docs.aws.amazon.com/ElasticLoadBalancing/latest/DeveloperGuide/access-log-collection.html#access-log-entry-format


