package io.github.adrianulbona

import org.apache.spark.ml.clustering.{KMeansModel, KMeans}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors.dense
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Yelp Spark ML
  *
  */
object SparkMLApp {

  def main(args: Array[String]) {
    val sqlContext: SQLContext = initSpark

    injectData(sqlContext)

    // 1. Cluster users based on the involvement within Yelp - the number of reviews is such a feature.
    // Play with the number of clusters. 3 clusters: beginners/regular/experts
    behaviourClusterUsers(sqlContext)

    // TODO - 1. As the city name is not consistent and instead of 10 city names we have many more, it will
    // be nice to group businesses based on their location
    // geoClusterBusinesses(sqlContext)

    sqlContext.sparkContext.stop()
  }

  def initSpark: SQLContext = {
    val conf = new SparkConf().setAppName("SparkML")
    conf.setMaster("local[*]")

    val sc: SparkContext = new SparkContext(conf)
    new SQLContext(sc)
  }

  def injectData(sqlContext: SQLContext) = {
    val businesses = sqlContext.read.json("data/yelp_academic_dataset_business.json")
    val users = sqlContext.read.json("data/yelp_academic_dataset_user.json")
    businesses.sample(withReplacement = false, 0.1).registerTempTable("business")
    users.registerTempTable("user")

  }

  def behaviourClusterUsers(sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._
    val dataset: DataFrame = sqlContext
      .sql("select user_id as id, review_count from user")
      .map({ case Row(id: String, reviewCount: Long) => Entity(id, dense(reviewCount.toDouble)) })
      .toDF()

    val kMeans = new KMeans()
      .setK(10)
      .setMaxIter(20)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")

    val userClusteringModel = kMeans.fit(dataset)
    userClusteringModel.clusterCenters.map(center => center(0)).foreach(println)

    // TODO - How the number of cluster is affecting the clustering quality?
    println(userClusteringModel.computeCost(dataset))

    userClusteringModel.transform(dataset)
  }

  def geoClusterBusinesses(sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._
    // extract latitude and longitude
    val dataset: DataFrame = ???

    // build the model
    val kMeans: KMeansModel = ???

    // train the model
    val localizedEntitiesKMeansModel: KMeansModel = ???
    localizedEntitiesKMeansModel.clusterCenters.map(center => center(0) + ", " + center(1)).foreach(println)

    // cluster all businesses and return the resulted dataset
    localizedEntitiesKMeansModel.transform(dataset)
  }


  case class Entity(id: String, features: Vector)
}
