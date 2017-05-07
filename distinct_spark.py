'''
Saurabh Mahajan
sm6921
Count the number of distinct words in a text file
Run the code using spark-submit distinct_spark.py
'''

from pyspark import SparkContext
import re

# remove any non-words, split into separate words and convert to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':

    sc = SparkContext("local", "distinctwords")
    # get the text from the file
    text = sc.textFile('pg2701.txt')
    # get the number of distinct words by splitting the text into words
    distinct_words = text.flatMap(splitter).distinct().count()
    
    print("Number of distinct words in the text are", distinct_words)