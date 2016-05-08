package io.github.adrianulbona

import org.apache.spark.ml.clustering.{KMeansModel, KMeans}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.Vectors.dense
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Yelp Recommender KMeans
  *
  */
object SparkMLApp {

  def main(args: Array[String]) {
    val sqlContext: SQLContext = initSpark

    val (businesses: DataFrame, reviews: DataFrame, users: DataFrame) = loadData(sqlContext)
    registerTempTables(businesses, reviews, users)

    // as the city name is not consistent and instead of 10 city names we have much more we must group somehow
    // businesses based from the same geographic area. why not using clustering for that!?
    val clusteredLocalizedEntities: DataFrame = geoClusterBusinesses(sqlContext)
    clusteredLocalizedEntities.registerTempTable("cbusiness")

    // 1. having the businesses clustered based on their location we will now focus on one city where for all users with
    // a decent amount of reviews we will try to find groups of users with similar tastes.
    // 2. entities -> users
    // 3. features -> businesses
    // 4. if an user gives an positive review to a business we mark this as an 1 within a sparse vector associated with
    // that user
    // 5. running kmeans
    val dataset: DataFrame = findSimilarUsers(sqlContext)._2

    dataset.groupBy("prediction").count().take(10).foreach(println)

    sqlContext.sparkContext.stop()
  }

  def initSpark: SQLContext = {
    val conf = new SparkConf().setAppName("Hello Spark")
    conf.setMaster("local[*]")

    val sc: SparkContext = new SparkContext(conf)
    new SQLContext(sc)
  }

  def loadData(sqlContext: SQLContext): (DataFrame, DataFrame, DataFrame) = {
    val businesses = sqlContext.read.json("data/yelp_academic_dataset_business.json")
    val reviews = sqlContext.read.json("data/yelp_academic_dataset_review.json")
    val users = sqlContext.read.json("data/yelp_academic_dataset_user.json")
    (businesses, reviews, users)
  }

  def registerTempTables(businesses: DataFrame, reviews: DataFrame, users: DataFrame): Unit = {
    businesses.registerTempTable("businesses")
    reviews.registerTempTable("reviews")
    users.registerTempTable("users")
  }

  def geoClusterBusinesses(sqlContext: SQLContext): DataFrame = {
    val places: DataFrame = sqlContext.sql("select business_id as id, latitude, longitude from businesses")
    import sqlContext.implicits._
    val localizedEntities: DataFrame = places
      .map({ case Row(id: String, lat: Double, long: Double) => Entity(id, dense(lat, long)) })
      .toDF()

    val kMeans = new KMeans()
      .setK(10)
      .setMaxIter(20)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
    val localizedEntitiesKMeansModel = kMeans.fit(localizedEntities)
    localizedEntitiesKMeansModel.clusterCenters.map(center => center(0) + ", " + center(1)).foreach(println)

    val clusteredLocalizedEntities: DataFrame = localizedEntitiesKMeansModel.transform(localizedEntities)
    clusteredLocalizedEntities
  }

  def findSimilarUsers(sqlContext: SQLContext): (KMeansModel, DataFrame) = {
    import sqlContext.implicits._
    val businessesFromGeoCluster0: DataFrame = sqlContext
      .sql("select b.business_id " +
        "from businesses b join cbusiness cb on b.business_id = cb.id " +
        "where cb.prediction = 0 and b.review_count > 30")
    businessesFromGeoCluster0.registerTempTable("someBusinesses")

    val reviewsFromRelevantUsers: DataFrame = sqlContext
      .sql("select r.business_id, r.user_id, r.stars " +
        "from reviews r join users u on r.user_id = u.user_id " +
        "where u.review_count > 30")
    reviewsFromRelevantUsers.registerTempTable("reviewsFromRelevantUsers")

    val positiveReviewsFromRelevanUsers: DataFrame = sqlContext
      .sql("select b.business_id, r.user_id " +
        "from someBusinesses b join reviewsFromRelevantUsers r on r.business_id = b.business_id " +
        "where r.stars = 5")
      .cache

    val dataset = positiveReviewsFromRelevanUsers.cache()

    val indexer = new StringIndexer()
      .setInputCol("business_id")
      .setOutputCol("bid")
      .fit(positiveReviewsFromRelevanUsers)

    val indexedDataset = indexer.transform(dataset)
    val numberOfDistinctBusinesses: Long = indexedDataset.select("bid").distinct().count()
    val ds = indexedDataset
      .map({ case Row(bid: String, uid: String, bidindexed: Double) => (uid, bidindexed) })
      .groupByKey()
      .map(userFav => {
        val uid = userFav._1
        val places: List[Double] = userFav._2.toList
        val toSeq: Seq[(Int, Double)] = places.distinct.map(pid => (pid.toInt, 1.0)).toSeq
        Entity(uid, Vectors.sparse(numberOfDistinctBusinesses.toInt, toSeq))
      }).toDF //.sample(withReplacement = false, 0.1)

    val kmeans: KMeans = ???
    val model: KMeansModel = ???
    val clusteredDataset: DataFrame = ???
    (model, clusteredDataset)
  }

  case class Entity(id: String, features: Vector)
}