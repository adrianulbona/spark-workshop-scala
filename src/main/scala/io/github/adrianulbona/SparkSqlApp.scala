package io.github.adrianulbona

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Yelp Spark SQL
  *
  */
object SparkSqlApp {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SparkBasic")
    conf.setMaster("local[*]")

    val sc: SparkContext = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sc)

    // 1. Map JSON files to RDDs
    val businessDF = sqlContext.read.json("data/yelp_academic_dataset_business.json")
    businessDF.sample(withReplacement = false, 0.01).registerTempTable("business")

    val checkInDF = sqlContext.read.json("data/yelp_academic_dataset_checkin.json")
    checkInDF.sample(withReplacement = false, 0.01).registerTempTable("checkin")

    val reviewDF = sqlContext.read.json("data/yelp_academic_dataset_review.json")
    reviewDF.sample(withReplacement = false, 0.01).registerTempTable("review")

    val userDF = sqlContext.read.json("data/yelp_academic_dataset_user.json")
    userDF.sample(withReplacement = false, 0.01).registerTempTable("user")

    // 2.1 How many entries contains the business RDD
    sqlContext.sql("select count(*) from business").show()
    // TODO - 2.1 Print the number of records for the others RDDs

    // 2.2 Find out the number of businesses from each city.
    sqlContext.sql("select city, count(*) from business group by city").show()

    // TODO - 2.2 Which are the top 10 cities based on number of businesses ?
    sqlContext.sql("select city, count(*) as c from business group by city order by c desc").show()

    // 2.3 Find our number of reviews from each day
    sqlContext.sql("select date, count(*) from review group by date")
    // TODO - 2.3 Which are the top 10 days based on number of reviews?
    // TODO - 2.3 What is the number of reviews for each star?

    // 2.4 Find out the first registered user
    sqlContext.sql("select * from user order by yelping_since desc limit 1").show()
    // TODO - 2.4 In which months was registered the most users ?

    // 2.5 Retrieve all the reviews for each business name
    sqlContext.sql("select b.name, r.text from business b join review r on b.business_id = r.business_id").show()
    // TODO - 2.5 Retrieve all the reviews of “Red White & Brew” business
    // TODO - 2.5 Retrieve all the checkins for each business

    sc.stop()
  }
}

