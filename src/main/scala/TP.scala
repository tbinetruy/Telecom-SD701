/*
 TP.scala

 sbt package && spark-submit target/scala-2.11/simple-project_2.11-1.0.jar
 */
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline

object TP {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder.appName("Simple Application").getOrCreate()

    val trainData = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("train-set.csv")

    val testData = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("test-set.csv")

    val trainCount = trainData.count()
    val trainCols = trainData.columns.length
    val testCount = testData.count()
    val testCols = testData.columns.length
    println(s"Train data size: $trainCount rows, $trainCols columns")
    println(s"Test data size: $testCount rows, $testCols columns")
    val inputcols = Array("Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(inputcols)
      .setOutputCol("features")

    val classifier = new LogisticRegression()
      .setLabelCol("Cover_Type")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(vectorAssembler, classifier))

    val model = pipeline.fit(trainData)

    var predictions = model.transform(testData)

    predictions = predictions.withColumn(
      "Cover_Type",
      predictions.col("prediction").cast(types.IntegerType)
    )
    predictions.printSchema()

    predictions.select("prediction", "Cover_Type").show()

    predictions.repartition(1).select("Id", "Cover_Type").write.option("header", "true").mode("overwrite").csv("results")


    spark.stop()
  }
}
