from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel




class BooksRS:

    def __init__(self, sc,modello):

        self.sc = sc
        if(modello==None):


            ratingLoaded = self.sc.textFile('ratings2.csv')
            header = ratingLoaded.take(1)[0]
            self.ratings = ratingLoaded.filter(lambda line: line != header) \
                .map(lambda line: line.split(",")).map(
                lambda fields: (int(fields[0]), int(fields[1]), float(fields[2]))).cache()
            # Load books


            booksLoaded = self.sc.textFile('boks2.csv')
            headerb = booksLoaded.take(1)[0]
            books = booksLoaded.filter(lambda line: line != headerb) \
                .map(lambda line: line.split(",")).map(lambda fields: (int(fields[0]), fields[1], fields[2])).cache()
            self.titles = books.map(lambda x: (int(x[0]), x[1])).cache()
            self.train2(self.ratings)
            self.model.save(self.sc, 'Modello2')
        else:

            self.model = MatrixFactorizationModel.load(self.sc, modello)


            ratingLoaded = self.sc.textFile('ratings2.csv')
            header = ratingLoaded.take(1)[0]
            self.ratings = ratingLoaded.filter(lambda line: line != header) \
                .map(lambda line: line.split(",")).map(
                lambda fields: (int(fields[0]), int(fields[1]), float(fields[2]))).cache()


            booksLoaded = self.sc.textFile('books2.csv')
            headerb = booksLoaded.take(1)[0]
            books = booksLoaded.filter(lambda line: line != headerb) \
                .map(lambda line: line.split(",")).map(lambda fields: (int(fields[0]), fields[1], fields[2])).cache()
            self.titles = books.map(lambda x: (int(x[0]), x[1])).cache()
    def ratingAVG(self):

        book = self.ratings.map(lambda x: (x[1], x[2])).groupByKey()
        book2 = book.map(lambda var:(var[0], (len(var[1]), float(sum(x for x in var[1])) / len(var[1]))))
        self.bookRateCount = book2.map(lambda x: (x[0], x[1][0]))


    def train2(self, ratings_RDD):

        rank = 8
        seed = 5
        iterations = 10
        regularization_parameter = 0.1

        self.model = ALS.train(ratings_RDD, rank, seed=seed,iterations=iterations, lambda_=regularization_parameter)



    def prediction(self, coupleUB):

        predict = self.model.predictAll(coupleUB)
        predicteR = predict.map(lambda x: (x.product, x.rating))
        self.ratingAVG()
        predicteR =predicteR.join(self.titles).join(self.bookRateCount)
        predicteR =predicteR.map(lambda pred: (pred[1][0][1], pred[1][0][0], pred[1][1]))

        return predicteR

    def newRating(self, ratings):

        ratings2 = self.sc.parallelize(ratings)
        self.ratings = self.ratings.union(ratings2)
        self.ratingAVG()
        self.train2()

        return ratings

    def getRatings(self, userID, booksId):
        book = self.sc.parallelize(booksId).map(lambda boook: (userID, boook))
        ratings = self.prediction(book).collect()

        return ratings

    def getTopK(self, userId, k):

        unratedBooks = self.ratings.filter(lambda rating: not rating[0] == userId).map(lambda book: (userId, book[1])).distinct()
        ratings = self.prediction(unratedBooks).filter(lambda rating: rating[2] > 20).takeOrdered(k, key=lambda book: -book[1])
        return ratings





