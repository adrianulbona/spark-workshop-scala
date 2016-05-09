package io.github.adrianulbona

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Yelp Spark RDD
  */
object SparkBasicApp {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SparkBasic")
    conf.setMaster("local[*]")

    val sc: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sc)

    val businessRDD = sqlContext.read.json("data/yelp_academic_dataset_business.json")
      .map(row => new Business(
        row.getAs[String]("business_id"),
        row.getAs[String]("city"),
        row.getAs[Long]("review_count"),
        row.getAs[String]("state"),
        row.getAs[String]("name"))
      )

    businessRDD.cache()

    // 1. Map JSON files to RDDs
    val checkinRDD = sqlContext.read.json("data/yelp_academic_dataset_checkin.json")
      .map(row => new CheckIn(
        row.getAs[String]("type"),
        row.getAs[String]("business_id"))
      )

    checkinRDD.cache()

    val reviewRDD = sqlContext.read.json("data/yelp_academic_dataset_review.json")
      .map(row => new Review(
        row.getAs[String]("review_id"),
        row.getAs[String]("date"),
        row.getAs[String]("business_id"),
        row.getAs[Long]("stars"))
      )

    reviewRDD.cache()

    val userRDD = sqlContext.read.json("data/yelp_academic_dataset_user.json")
      .map(row => new User(
        row.getAs[String]("yelping_since"),
        row.getAs[Long]("review_count"),
        row.getAs[String]("name"),
        row.getAs[String]("user_id"),
        row.getAs[Long]("fans"))
      )

    userRDD.cache()

    // 2.1 How many entries contains the business RDD
    println("Number of businesses: " + businessRDD.count())

    // TODO - 2.1 Print the number of records for the others RDDs

    // 2.2 Find out the number of businesses from each city.
    val businessesFromCities: RDD[(String, Iterable[Business])] = businessRDD.groupBy(row => row.city.trim())
    val topCities: Array[(String, Int)] = businessesFromCities.take(10).map(row => (row._1, row._2.toList.length))
    topCities.foreach(element => println(element._1, element._2))

    // TODO - 2.2 Which are the top 10 cities based on number of businesses ?

    // 2.3 Find our number of reviews from each day
    reviewRDD.groupBy(row => row.date)
      .map(row => (row._1, row._2.toList.length))
      .take(10)
      .foreach(row => println(row._1, row._2))

    // TODO - 2.3 Which are the top 10 days based on number of reviews?
    // TODO - 2.3 What is the number of reviews for each star?

    // 2.4 Find out the first registered user
    val firstUser: User = userRDD.sortBy(_.yelpingSince, ascending = false).first()
    println(firstUser)

    // TODO - 2.4 In which months was registered the most users ?

    // 2.5 Retrieve all the reviews for each business
    val groupedReviewsRDD: RDD[(String, Iterable[Review])] = reviewRDD.groupBy(review => review.businessId)
    val businessNamesRDD: RDD[(String, String)] = businessRDD.map(business => (business.businessId, business.name))
    val reviewsGroupedByBusinessNames: RDD[(String, (String, Iterable[Review]))] = businessNamesRDD.join(groupedReviewsRDD)
    reviewsGroupedByBusinessNames.take(10).foreach(element => println(element._2))

    // TODO - 2.5 Retrieve all the reviews of “Red White & Brew” business
    // TODO - 2.5 Retrieve all the checkins for each business

    sc.stop()
  }

  case class Business(businessId: String, city: String, reviewCount: Long, state: String, name: String)
  case class CheckIn(chType: String, businessId: String)
  case class Review(reviewId: String, date: String, businessId: String, stars: Long)
  case class User(yelpingSince: String, reviewCount: Long, name: String, userId: String, fans: Long)
}

