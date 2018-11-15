/*

 TP.scala

 sbt package && spark-submit target/scala-2.11/simple-project_2.11-1.0.jar
 */

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, types, DataFrame, Row, Dataset}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.classification.{
  LogisticRegression,
  RandomForestClassifier,
  GBTClassifier,
  DecisionTreeClassifier,
  LinearSVC
}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.ml.feature.{
  PCA,
  VectorIndexer,
  IndexToString,
  StringIndexer
}

import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.{
  MulticlassClassificationEvaluator,
  RegressionEvaluator,
  BinaryClassificationEvaluator
}

import org.apache.spark.ml.param.{
  DoubleParam,
  IntParam,
  ParamMap
}

object TP {
  def fetchTrainData(spark: SparkSession): DataFrame = {
    return spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("../../train-set.csv")
  }
  def fetchTestData(spark: SparkSession): DataFrame = {
    return spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("../../test-set.csv")
  }
  def describeResult(result: DataFrame) = {
    result
      .groupBy("Cover_Type", "prediction").count.show()

  }
  def describe(trainData: DataFrame, testData: DataFrame) = {
    val trainCount = trainData.count()
    val trainCols = trainData.columns.length
    val testCount = testData.count()
    val testCols = testData.columns.length
    println(s"Train data size: $trainCount rows, $trainCols columns")
    println(s"Test data size: $testCount rows, $testCols columns")
  }
  def getInputCols(): Array[String] = {
    return Array("Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40")
  }
  def saveToCsv(result: DataFrame) = {
    result
      .repartition(1)
      .withColumn(
        "Cover_Type",
        result.col("prediction").cast(types.IntegerType))
      .select("Id", "Cover_Type")
      .write.option("header", "true")
      .mode("overwrite")
      .csv("results")
  }
  def getPca(): PCA = {
    return new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(10)
  }
  def getCrossValidator(pipeline: Pipeline, paramGrid: Array[ParamMap], numFolds: Int): CrossValidator = {
    return new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("Cover_Type"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)  // Use 3+ in practice
  }
  def getRandomForestModel(): CrossValidator = {
    val vectorAssembler = this.getVectorAssembler()

    val classifier = new RandomForestClassifier()
      .setLabelCol("Cover_Type")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(vectorAssembler, classifier))

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.numTrees, Array(10, 50, 100))
      .addGrid(classifier.maxDepth, Array(1, 5, 10))
      .build()

    return this.getCrossValidator(pipeline, paramGrid, 2)
  }
  def getDecisionTree(): CrossValidator = {
    val vectorAssembler = this.getVectorAssembler()

    val classifier = new DecisionTreeClassifier()
      .setLabelCol("Cover_Type")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(vectorAssembler, classifier))

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxDepth, Array(29))
      .build()

    return this.getCrossValidator(pipeline, paramGrid, 6)
  }
  def getLogisticRegModel(): CrossValidator = {
    val vectorAssembler = this.getVectorAssembler()

    val pca = this.getPca()

    val classifier = new LogisticRegression()
      .setLabelCol("Cover_Type")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(vectorAssembler, classifier))

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.regParam, Array(0.1))
      .build()

    return this.getCrossValidator(pipeline, paramGrid, 3)
  }
  def getVectorAssembler(): VectorAssembler = {
    return new VectorAssembler()
      .setInputCols(this.getInputCols)
      .setOutputCol("features")
  }
  def score(cv: CrossValidator, trainData: DataFrame): DataFrame = {
    val Array(training, test) = trainData.randomSplit(Array(0.7, 0.3), seed = 12345)

    val cvModel = cv.fit(training)
    val result = cvModel.transform(test)

    val score = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
      .evaluate(result)

    println("\n\n=============\n\n f1 score for model: " + score)
    println("\n\n=============\n\n")

    import java.io.PrintWriter
    new PrintWriter("f1-score") { write(score.toString()); close }

    return result
  }
  def main(args: Array[String]) {
    val sc = SparkContext.getOrCreate()
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val spark = SparkSession
      .builder.appName("Simple Application").getOrCreate()

    var trainData = this.fetchTrainData(spark)
    var testData = this.fetchTestData(spark)

    this.describe(trainData, testData)

    var pipeline: CrossValidator = new CrossValidator()
    Integer.parseInt(args(0)) match {
      case 0 => pipeline = this.getLogisticRegModel()
      case 1 => pipeline = this.getRandomForestModel()
      case 2 => pipeline = this.getDecisionTree()
    }

    val result = this.score(pipeline, trainData)

    this.saveToCsv(result)
    this.describeResult(result)

    result.show()

    spark.stop()
  }
}
