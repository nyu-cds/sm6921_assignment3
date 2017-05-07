'''
Saurabh Mahajan
sm6921
Spark program to compute the average of square roots of numbers from 1 to 1000
Run the code using spark-submit squareroot_spark.py
'''
from pyspark import SparkContext

sc = SparkContext("local", "averagesquareroots")

# create an RDD of numbers from 1 to 1000
nums = sc.parallelize(range(1, 1001))

# map all values to their square roots
sq_roots = nums.map(lambda a: a ** 0.5)

# calculate average of roots by adding all the square roots computed in last step
roots_average = sq_roots.fold(0, lambda a,b:a+b) / sq_roots.count()

print("Average of square roots of first 1000 numbers", roots_average)