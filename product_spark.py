'''
Saurabh Mahajan
sm6921
Spark program to compute the product of numbers from 1 to 1000
Run the code using spark-submit product_spark.py
'''

from pyspark import SparkContext

sc = SparkContext("local", "product")

# create an RDD of numbers from 1 to 1000
nums = sc.parallelize(range(1, 1001))

# calculate product of all the numbers from 1 to 1000
product = nums.fold(1, lambda a,b:a*b)

print("Product of first 1000 natural numbers", product)