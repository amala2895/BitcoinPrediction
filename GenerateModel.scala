/* Generation of Prediction Model for predicting Bitcoin price
based on number of tweets and number of google searches two days ago.
This model user Linear Regression to correlate the features
 */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD


object GenerateModel {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("GenerateModel");

    val sc = new SparkContext(conf);

    val spark = SparkSession
      .builder()
      .appName("GenerateModel")
      .config("spark.some.config.option", "some-value")
      .getOrCreate();


    /*Load the cleaned and joined data. The columns in the loaded dataset  are
    Bitcoin Price, Number of Tweets, Number of Google Searches
    about Bitcoin */
    /*PATH TO THE INPUT DATA */
    /* Number of elements in the data set */
    val rawDataRDD = sc.textFile("/tmp/BDAD-Project/InputData.csv")

    val NumOfDataPoints = rawDataRDD.count();


    /*Split the RDD into two RDDs (Testing and Training Data with 80 percent being
    *training data and the remaining data is for testing the model)*/
    val splitTrainingAndTesting = rawDataRDD.randomSplit(Array(0.8,0.2),2);


    /*Parse the testing and testing data to generate Labeled data points
    * The data points have the relation in the form
    * y = w1 * x1 + w2 * x2 + c where y represents Bitcoin price
    * x1 represents Number of tweets for related to Bitcoin
    * and w1 is the weight that will be given to x1 by the model
    * generation process.
    * x2 represents Number of Google searches about Bitcoin and w2
    * is the weight that will be given to x2 by the model
    * generation process. c represents the intercept that will also
    * be calculated by the model generation process*/

    val parsedTestingRDD = splitTrainingAndTesting(0).map {line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble , Vectors.dense(parts(1).toDouble, parts(2).toDouble))
    }.cache() ;


    val parsedTrainingRDD = splitTrainingAndTesting(1).map {line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble , Vectors.dense(parts(1).toDouble , parts(2).toDouble ))
    }.cache() ;


    val cnt = parsedTestingRDD.count();
    println("Count of TestingRDD is " + cnt);

    parsedTestingRDD.foreach(println);

    /* Parameters required for model generation */
    val numIterations = 100
    val stepSize = 0.0001

    /* Model definition */
    val algorithm = new LinearRegressionWithSGD()
    algorithm.setIntercept(true)
    algorithm.optimizer
      .setNumIterations(numIterations)
      .setStepSize(stepSize)

    /* Generate the model */
    val model = algorithm.run(parsedTestingRDD);

    /* Weights and intercept of the generated model */
    println("weights: %s, intercept: %s".format(model.weights, model.intercept));

    println("The MODEL is::  " + model);

    /* Evaluate model on training examples and compute training error */
    val valuesAndPreds = parsedTrainingRDD.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    /*Print the predicted values */
   // valuesAndPreds.foreach(println);

    /* MSE gives an idea about the accuracy of the model */
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((p - v), 2) }.mean()

    // Save the prediction model
    model.save(sc,"/tmp/BDAD-Project/scalaLinearRegressionWithSGDModel");

  }
}
