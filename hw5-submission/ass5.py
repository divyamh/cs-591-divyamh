from pyspark.mllib.recommendation import ALS
from pyspark import SparkContext
sc =SparkContext()
 
# load training and test data into (user, product, rating) tuples


def parseRating(line):
    fields = line.split(',')
    #user,item = line['reviewerID'],line['asin']
    #ratings = (line['overall'])
    return (int(fields[1]), int(fields[2]), float(fields[3]))   
def parseRating2(line):
    fields = line.split(',')
    #user,item = line['reviewerID'],line['asin']
    #ratings = (line['overall'])
    return (int(fields[1]), int(fields[2]))  
training = sc.textFile("traindata.csv").map(parseRating).cache()
test = sc.textFile("testdata.csv").map(parseRating2)
 
# train a recommendation model
model = ALS.train(training, rank = 10, iterations = 10)
 
# make predictions on (user, product) pairs from the test data
predictions = model.predictAll(test.map(lambda x: (x[0], x[1])))
predictions.saveAsTextFile("result")
