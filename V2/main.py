from engine import *

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math

sc=SparkContext("local","job1")

b=RecommendationEngine(sc,'Modello2')
c=b.get_top_ratings(3,5)
print (c)