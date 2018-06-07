from Recommender import *

from pyspark import SparkContext


sc=SparkContext("local","job1")

b=BooksRS(sc, 'Modello2')
c=b.getTopK(3, 5)
print (c)