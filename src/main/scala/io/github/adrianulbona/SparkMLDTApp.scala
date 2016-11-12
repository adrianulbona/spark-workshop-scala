package io.github.adrianulbona

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Yelp Spark ML
  *
  */
object SparkMLDTApp {

  def main(args: Array[String]) {
    val sqlContext: SQLContext = initSpark

    val locationDataFrame: DataFrame = buildLocationDataframe(sqlContext)

    classifyLocations(locationDataFrame)

    sqlContext.sparkContext.stop
  }

  def initSpark: SQLContext = {
    val conf = new SparkConf().setAppName("SparkML")
    conf.setMaster("local[*]")

    val sc: SparkContext = new SparkContext(conf)
    new SQLContext(sc)
  }

  def buildLocationDataframe(sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._
    // needed for $ and other implicit conversions

    // load initial data and filter only that is of interest
    val businesses = sqlContext.read.json("data/yelp_academic_dataset_business.json")
    val cityLatLong = businesses.select($"city", $"latitude", $"longitude")

    // find some representative cities
    val someCities = cityLatLong.groupBy($"city").count.orderBy($"count".desc).select($"city").as[String].take(5)

    // define a Spark ML specific converter as UDF
    val latLongVector = udf { (lat: Double, long: Double) => Vectors.dense(lat, long) }

    // build the actual Spark ML dataset: features = lat and long, label = city name
    cityLatLong
      .filter($"city".isin(someCities: _*))
      .select(latLongVector($"latitude", $"longitude").as("features"), $"city".as("label"))

  }

  def classifyLocations(locationDataFrame: DataFrame) = {
    // city names => numeric
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(locationDataFrame)

    // classifier definition
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")

    // numeric => city names
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // glue everything
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, dt, labelConverter))

    // keep some data for testing
    val Array(trainingData, testData) = locationDataFrame.randomSplit(Array(0.7, 0.3))

    // train
    val model = pipeline.fit(trainingData)

    // predict
    val predictions = model.transform(testData)

    // show some predictions
    predictions.select("predictedLabel", "label", "features").show(5)

    // compute accuracy
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    // visualize the actual decision tree
    val treeModel = model.stages(1).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)
  }
}
