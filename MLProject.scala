// Databricks notebook source
//rishabhbhardwaj/spark-timeseries is licensed under the
//MIT License


// COMMAND ----------

import breeze.linalg._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import spark.implicits._
import Numeric.Implicits._

// COMMAND ----------

//I took the top 10 competitors in Apple's competitive set are Microsoft, Oracle, IBM, HP, SAP, Sony, Samsung, Dell, Google and Salesforce and include their prices to observation. I wanted to see a market cycles and diffrence between competitors.

// COMMAND ----------

var df = spark.sql("""SELECT date, sum(appl) appl, sum(volume) volume, sum(sp500) sp500, sum(microsoft) microsoft, sum(oracle) oracle, sum(ibm) ibm, sum(hp) hp, sum(sap) sap, sum(sony) sony, sum(samsung) samsung, sum(dell) dell, sum(googl) googl, sum(salesforce) salesforce  
FROM(SELECT date date, close appl, volume volume, 0 sp500, 0 microsoft, 0 oracle, 0 ibm, 0 hp, 0 sap, 0 sony, 0 samsung, 0 dell, 0 googl, 0 salesforce from AAPL_CSV
UNION ALL
SELECT date, 0, 0, close, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 FROM gspc WHERE year(date) > 1979
UNION ALL
SELECT date, 0, 0, 0, close, 0, 0, 0, 0, 0, 0, 0, 0, 0 FROM microsoft
UNION ALL
SELECT date, 0, 0, 0, 0, close, 0, 0, 0, 0, 0, 0, 0, 0 FROM oracle
UNION ALL
SELECT date, 0, 0, 0, 0, 0, close, 0, 0, 0, 0, 0, 0, 0 FROM ibm
UNION ALL
SELECT date, 0, 0, 0, 0, 0, 0, close, 0, 0, 0, 0, 0, 0 FROM hp WHERE year(date) > 1979
UNION ALL
SELECT date, 0, 0, 0, 0, 0, 0, 0, close, 0, 0, 0, 0, 0 FROM sap
UNION ALL
SELECT date, 0, 0, 0, 0, 0, 0, 0, 0, close, 0, 0, 0, 0 FROM sony WHERE year(date) > 1979
UNION ALL
SELECT date, 0, 0, 0, 0, 0, 0, 0, 0, 0, close, 0, 0, 0 FROM samsung
UNION ALL
SELECT date, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, close, 0, 0 FROM dell
UNION ALL
SELECT date, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, close, 0 FROM googl
UNION ALL
SELECT date, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,close FROM salesforce) GROUP BY date""")
df.show()

// COMMAND ----------

//There are about 10k rows into our data frame. To optimize the view, i added "years' and 'months' to observation.

// COMMAND ----------

df = df.withColumn("year", year(col("date")))
var df_ = df.groupBy($"year").agg(mean($"appl"),mean($"volume"),mean($"sp500"),mean($"microsoft"),mean($"oracle"),mean($"ibm"),mean($"hp"),mean($"sap"),mean($"sony"),mean($"samsung"),mean($"dell"),
                              mean($"googl"),mean($"salesforce"))
df_.printSchema()

// COMMAND ----------

df.describe("appl","sp500","microsoft","oracle","ibm","hp").show()
df.describe("sap","sony","samsung","dell","googl","salesforce").show()

// COMMAND ----------

//The covariance is sometimes called a measure of "linear dependence" between the two random variables. 
//When the covariance is normalized, one obtains the Pearson correlation coefficient, which gives the goodness of the fit for the best possible linear function describing the relation between the variables. In this sense covariance is a linear gauge of dependence.
//We see the tendency when the prise of sp500 up then appl prise up as well

// COMMAND ----------

println(s"median and quantiles appl (close price): ${ df.stat.approxQuantile("appl",
           Array(0.25,0.5,0.75),0.0)}")
println(s"correlation appl - volume (close price): ${df.stat.corr("appl","volume")}")
println(s"covariance appl - sp500 (close price): ${df.stat.cov("appl","sp500")}")
println(s"correlation appl - microsoft (close price): ${df.stat.corr("appl","microsoft")}")
println(s"correlation appl - oracle (close price): ${df.stat.corr("appl","oracle")}")
println(s"correlation appl - ibm (close price): ${df.stat.corr("appl","ibm")}")
println(s"correlation appl - hp (close price): ${df.stat.corr("appl","hp")}")
println(s"correlation appl - sap (close price): ${df.stat.corr("appl","sap")}")
println(s"correlation appl - sony (close price): ${df.stat.corr("appl","sony")}")
println(s"correlation appl - samsung (close price): ${df.stat.corr("appl","samsung")}")
println(s"correlation appl - dell (close price): ${df.stat.corr("appl","dell")}")
println(s"correlation appl - googl (close price): ${df.stat.corr("appl","googl")}")
println(s"correlation appl - salesforce (close price): ${df.stat.corr("appl","salesforce")}")


// COMMAND ----------

//we see the market capitalizations of 500 large companies were developing much more than Apple - the line of last one much slow.//
display(df_)

// COMMAND ----------

//The tendency raising of the prise of tech companies the same during 1980-2018 years. The IBM price was raising  much more for 1999-2017 but then run to down.
display(df_)

// COMMAND ----------

//we could notice that Sony close price has еру peak at the 2008 year and has slow tendency to grow. The Apple price started much more grow from 2005 year to now. The same the Salesforse.
display(df_)

// COMMAND ----------

//Microsoft and Oracle have the information of 1994-2018 years
display(df.select($"appl",$"sap",$"year").filter(($"year">1994)&&($"year"<2019)))

// COMMAND ----------

//Microsoft and Oracle have the information of 1986-2018 years
display(df.select($"appl",$"microsoft",$"oracle",$"year").filter(($"year">1985)&&($"year"<2019)))

// COMMAND ----------

//Dell has the information for last three years - 2016-2018
display(df.select($"appl",$"dell", $"year").filter(($"year">2015)&&($"year"<2019)))

// COMMAND ----------

//salesforce
display(df.select($"appl",$"salesforce",$"year").filter(($"year">2004)&&($"year"<2019)))

// COMMAND ----------

//we see the market of Google and Samsung were developing much more than Apple - the line of last one much slow. 
display(df.select($"appl",$"googl",$"year").filter(($"year">2004)&&($"year"<2019)))

// COMMAND ----------

display(df.select($"appl",$"samsung",$"year").filter(($"year">2004)&&($"year"<2019)))

// COMMAND ----------

//Let’s briefly discuss this. Open is the price of the stock at the beginning of the trading day (it need not be the closing price of the previous trading day), high is the highest price of the stock on that trading day, low the lowest price of the stock on that trading day, and close the price of the stock at closing time. Volume indicates how many stocks were traded. Adjusted close is the closing price of the stock that adjusts the price of the stock for corporate actions. While stock prices are considered to be set mostly by traders, stock splits (when the company makes each extant stock worth two and halves the price) and dividends (payout of company profits per share) also affect the price of a stock and should be accounted for.

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select * from appl_csv sort by Date

// COMMAND ----------

var df_div = spark.sql("select year(Date) year, month(Date) month, Date, Dividends from appl_csv sort by year, month")
df_div.show()

// COMMAND ----------

//In this window spec, the data is partitioned by dates. Each day’s data is ordered by date. And, the window frame is defined as starting from -4 (one row before the current row) and ending at 4 (one row after the current row), for a total of 9 rows in the sliding window.
// Create a window spec.

// PARTITION BY country ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
val windowSpec = Window.orderBy("Date")


// COMMAND ----------

//These predictions take several variables into account such as volume changes, price changes, market cycles, similar stocks.
// Calculate the stock_return
df_ = df.select($"date",$"appl",$"volume",$"date",$"year").filter(($"appl" =!= 0)||($"appl" =!= null)).withColumn("price_change", $"appl" - lag("appl",4).over(windowSpec))
df_ = df_.withColumn("volume_change", $"Volume" - lag("Volume",4).over(windowSpec))
df_.show()

// COMMAND ----------

df_ = df_.withColumn("month", month(col("date")))

// COMMAND ----------

val df_temp = df_.filter(($"price_change" =!= 0)||($"price_change" =!= null)).groupBy($"year",$"month").agg(mean($"price_change"),mean($"volume_change"))

// COMMAND ----------

display(df_temp)

// COMMAND ----------

display(df_temp)

// COMMAND ----------

display(df_temp.filter($"year" < 2004))

// COMMAND ----------

display(df_temp.filter($"year" > 2003))

// COMMAND ----------

display(df_temp.filter($"year" === 2018))

// COMMAND ----------

display(df_temp.filter($"year" === 2018))

// COMMAND ----------

//we could notice specific cycling  during the months
display(df_temp.filter($"year" < 1986))

// COMMAND ----------

display(df_temp.filter(($"year" > 1985)&&($"year" < 1991)))

// COMMAND ----------

display(df_temp.filter(($"year" > 1990)&&($"year" < 1996)))

// COMMAND ----------

display(df_temp.filter(($"year" > 1995)&&($"year" < 2001)))

// COMMAND ----------

display(df_temp.filter(($"year" > 2000)&&($"year" < 2006)))

// COMMAND ----------

display(df_temp.filter(($"year" > 2005)&&($"year" < 2011)))

// COMMAND ----------

display(df_temp.filter($"year" > 2010))

// COMMAND ----------

df_.filter($"year" < 1999).describe("appl","volume","price_change","volume_change").show
df_.filter(($"year" > 1998)&&($"year" < 2008)).describe("appl","volume","price_change","volume_change").show

// COMMAND ----------

df_.filter(($"year" > 2007)&&($"year" < 2012)).describe("appl","volume","price_change","volume_change").show
df_.filter($"year" > 2011).describe("appl","volume","price_change","volume_change").show

// COMMAND ----------

display(df_.groupBy($"year",$"month").mean("volume"))

// COMMAND ----------

display(df_.groupBy($"year",$"month").agg(mean("appl")))

// COMMAND ----------

df_ = df_.select($"date",$"appl",$"volume",$"price_change",$"volume_change",$"year")

// COMMAND ----------

df_ = df_.na.drop()

// COMMAND ----------

df_.printSchema()

// COMMAND ----------

import com.cloudera.sparkts.models.{ARIMA, ARIMAModel}
import org.apache.spark.mllib.linalg.Vectors

val dates = df_.collect().flatMap((row: Row) => Array(row.get(0)))
val amounts = df_.select("appl").map(r => r(0).asInstanceOf[Double]).collect()

val actual = Vectors.dense(amounts)
val model = ARIMA.autoFit(actual)
println("best-fit model ARIMA(" + model.p + "," + model.d + "," + model.q + ") AIC=" + model.approxAIC(actual))

// COMMAND ----------

val arimaModel_ = ARIMA.fitModel(1,2,1, actual) //d=2. q=1 does not work

// COMMAND ----------

//We can represent our model as ARIMA(ar-term, ma-term, i-term) 

// COMMAND ----------

val arimaModel = ARIMA.fitModel(1,2,0, actual) //d=2. q=1 does not work

// COMMAND ----------

val period = actual.size
val predicted = arimaModel.forecast(actual, period)

// COMMAND ----------

println("coefficients: " + arimaModel.coefficients.mkString(","))

// COMMAND ----------

var totalErrorSquare = 0.0
for (i <- 0 until period-1) {
  val errorSquare = Math.pow(predicted(i) - amounts(i), 2)
  println(dates(i) + ":\t\t" + predicted(i) + "\t should be \t" + amounts(i) + "\t Error Square = " + errorSquare)
  totalErrorSquare += errorSquare
}


// COMMAND ----------

println("Mean Square Error: " + totalErrorSquare/period)

// COMMAND ----------

//Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors. It is usually defined as follows: where At is the actual value and Ft is the forecast value.

// COMMAND ----------

var diff = 0.0
for (i <- 0 until period-1) {
  diff += (Math.abs(predicted(i) - amounts(i)) / ((Math.abs(amounts(i)) + Math.abs(predicted(i)))/2))
}
val sMAPE = diff/period*100
println(s"Test SMAPE (percentage): ${sMAPE}")

// COMMAND ----------

val DAYS = 5

val forecast = arimaModel.forecast(actual, DAYS)
println("forecast of next 5 observations: " + forecast.toArray.mkString(","))

// COMMAND ----------

//i want to get ARIMA for few features but i did not find how i would do it
import org.apache.spark.ml.feature.VectorAssembler

val featureassembler = new VectorAssembler().
  setInputCols(Array("price_change","volume_change","volume")).
  setOutputCol("features")

val finalized_data = featureassembler.transform(df_).select($"features",$"appl")

// COMMAND ----------

finalized_data.count

// COMMAND ----------

//D = In an ARIMA model we transform a time series into stationary one(series without trend or seasonality) using differencing. D refers to the number of differencing transformations required by the time series to get stationary.

//Stationary time series is when the mean and variance are constant over time. It is easier to predict when the series is stationary.

//Differencing is a method of transforming a non-stationary time series into a stationary one. This is an important step in preparing data to be used in an ARIMA model.

//The first differencing value is the difference between the current time period and the previous time period. If these values fail to revolve around a constant mean and variance then we find the second differencing using the values of the first differencing. We repeat this until we get a stationary series

//The best way to determine whether or not the series is sufficiently differenced is to plot the differenced series and check to see if there is a constant mean and variance.

//Q = This variable denotes the lag of the error component, where error component is a part of the time series not explained by trend or seasonality

// COMMAND ----------

// Split the data into training and test sets (20% held out for testing).
//val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))


// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
//import org.apache.spark.sql.{Encoders, SparkSession}
import org.apache.spark.ml.regression.GeneralizedLinearRegression

// COMMAND ----------

//If we look at the data we can see that the series is not stationary in both mean and variance wise. We need to do a transformation. here I used log10 transformation after differencing. 

// COMMAND ----------

display(df_.groupBy($"year").agg(mean($"price_change")))

// COMMAND ----------

df_ = df_.withColumn("price_change_log",  log10($"price_change"))

// COMMAND ----------

display(df_.groupBy($"year").agg(mean($"price_change_log"), mean($"price_change")))

// COMMAND ----------

display(df_.groupBy($"year").agg(mean($"volume_change")))

// COMMAND ----------

display(df_.filter($"year" < 1999))

// COMMAND ----------

df_ = df_.na.drop()

// COMMAND ----------

display(df_.groupBy($"year").agg(mean($"volume_change")))

// COMMAND ----------

df_ = df_.withColumn("volume_change_log",  log10($"volume_change"))

// COMMAND ----------

display(df_.groupBy($"year").agg(mean($"volume_change_log"), mean($"volume_change")))

// COMMAND ----------

df_ = df_.na.drop()

// COMMAND ----------

 display(df_.groupBy($"year").agg(mean($"appl")))

// COMMAND ----------

df_ = df_.withColumn("appl_log",  log10($"appl"))

// COMMAND ----------

 display(df_.groupBy($"year").agg(mean($"appl_log"), mean($"appl")))

// COMMAND ----------

df_ = df_.withColumn("volume_log",  log10($"volume"))

// COMMAND ----------

df_ = df_.na.drop()

// COMMAND ----------

//So what do we need to do to get our data ready for Machine Learning?

//Recall our goal: We want to learn to predict the count of how close price will be changed (the appl column). We refer to the count as our target "labelColumn".
//This will use the Gradient-Boosted Trees (GBT) algorithm to learn how to predict close price from the feature vectors.

// COMMAND ----------

df_.count

// COMMAND ----------

//We'll split the set into training and test data
val Array(trainingData, testData) = df_.randomSplit(Array(0.8, 0.2))
val labelColumn = "appl_log"
//We define the assembler to collect the columns into a new column with a single vector - "features"
val assembler = new VectorAssembler()
            .setInputCols(Array("volume_log","price_change_log", "volume_change_log"))
            .setOutputCol("features")
//For the regression we'll use the Gradient-boosted tree estimator
val gbt = new GBTRegressor()
            .setLabelCol(labelColumn)
            .setFeaturesCol("features")
            .setPredictionCol("Predicted " + labelColumn);
//We define the Array with the stages of the pipeline
val stages = Array(
            assembler,
            gbt)
//Construct the pipeline
val pipeline_GBT = new Pipeline().setStages(stages)
//We fit our DataFrame into the pipeline to generate a model
val model_GBT = pipeline_GBT.fit(trainingData)
//We'll make predictions using the model and the test data
val predictions_GBT = model_GBT.transform(testData)
//This will evaluate the error/deviation of the regression using the Root Mean Squared deviation
val evaluator_GBT = new RegressionEvaluator()
            .setLabelCol(labelColumn)
            .setPredictionCol("Predicted " + labelColumn)
            .setMetricName("mse")
//We compute the error using the evaluator
val error = evaluator_GBT.evaluate(predictions_GBT)
println(error)

// COMMAND ----------

predictions_GBT.columns

// COMMAND ----------

display(predictions_GBT.select("Predicted appl_log","appl_log","year"))

// COMMAND ----------

import org.apache.spark.mllib.evaluation.RegressionMetrics
val out = predictions_GBT
  .select("Predicted appl_log", "appl_log")
  .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
val metrics = new RegressionMetrics(out)
println(s"MSE = ${metrics.meanSquaredError}")
println(s"RMSE = ${metrics.rootMeanSquaredError}")
println(s"R-squared = ${metrics.r2}")
println(s"MAE = ${metrics.meanAbsoluteError}")
println(s"Explained variance = ${metrics.explainedVariance}")

// COMMAND ----------

//We'll split the set into training and test data
val Array(trainingData, testData) = df_.randomSplit(Array(0.8, 0.2))
val labelColumn_ = "appl"
//We define the assembler to collect the columns into a new column with a single vector - "features"
val assembler_ = new VectorAssembler()
            .setInputCols(Array("volume","price_change", "volume_change"))
            .setOutputCol("features")
//For the regression we'll use the Gradient-boosted tree estimator
val gbt_ = new GBTRegressor()
            .setLabelCol(labelColumn_)
            .setFeaturesCol("features")
            .setPredictionCol("Predicted " + labelColumn_);
val stages_ = Array(
            assembler_,
            gbt_)
//Construct the pipeline
val pipeline_GBT_ = new Pipeline().setStages(stages_)
//We fit our DataFrame into the pipeline to generate a model
val model_GBT_ = pipeline_GBT_.fit(trainingData)
//We'll make predictions using the model and the test data
val predictions_GBT_ = model_GBT_.transform(testData)
//This will evaluate the error/deviation of the regression using the Root Mean Squared deviation
val evaluator_GBT_ = new RegressionEvaluator()
            .setLabelCol(labelColumn_)
            .setPredictionCol("Predicted " + labelColumn_)
            .setMetricName("mse")
//We compute the error using the evaluator
val error_ = evaluator_GBT_.evaluate(predictions_GBT_)
println(error_)

// COMMAND ----------

val out = predictions_GBT_
  .select("Predicted appl", "appl")
  .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
val metrics_ = new RegressionMetrics(out)
println(s"MSE = ${metrics_.meanSquaredError}")
println(s"RMSE = ${metrics_.rootMeanSquaredError}")
println(s"R-squared = ${metrics_.r2}")
println(s"MAE = ${metrics_.meanAbsoluteError}")
println(s"Explained variance = ${metrics_.explainedVariance}")

// COMMAND ----------

//The GBT algorithm has several hyperparameters, and tuning them to our data can improve accuracy. We will do this tuning using Spark's Cross Validation framework, which automatically tests a grid of hyperparameters and chooses the best.

// COMMAND ----------

import org.apache.spark.ml.tuning.{ParamGridBuilder,CrossValidator}
val gbt_gr = new GBTRegressor()
            .setLabelCol(labelColumn)
            .setFeaturesCol("features")
            .setPredictionCol("Predicted " + labelColumn);


// COMMAND ----------

//to prevent overfit GBT we can use cross-validation

// COMMAND ----------

val paramGrid = new ParamGridBuilder()
  .addGrid(gbt_gr.maxDepth, Array(5, 10, 15))
  .addGrid(gbt_gr.impurity, Array("variance"))
  .addGrid(gbt_gr.maxBins, Array(25, 30, 35))
  .addGrid(gbt_gr.maxIter, Array(5 ,10, 15, 20, 25))
  .build()

// COMMAND ----------

gbt_gr.extractParamMap

// COMMAND ----------

//Construct the pipeline
val pipeline_GBT_gr = new Pipeline().setStages(stages)
val evaluator_GBT_gr = new RegressionEvaluator()
            .setLabelCol(labelColumn)
            .setPredictionCol("Predicted " + labelColumn)
            .setMetricName("mse")

// COMMAND ----------

val cv = new CrossValidator()
  .setEstimator(pipeline_GBT_gr)
  .setEvaluator(evaluator_GBT_gr)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)  

// COMMAND ----------

// Run cross-validation, and choose the best set of parameters.
val cvModel = cv.fit(trainingData)

// COMMAND ----------

val prediction_gr = cvModel.bestModel.transform(testData)
val out_gr = prediction_gr
  .select("Predicted appl_log", "appl_log")
  .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
val metrics_gr = new RegressionMetrics(out_gr)
println(s"training data = ${trainingData.count}")
println(s"test data = ${testData.count}")
println(s"=====================================")
println(s"MSE = ${metrics_gr.meanSquaredError}")
println(s"RMSE = ${metrics_gr.rootMeanSquaredError}")
println(s"R-squared = ${metrics_gr.r2}")
println(s"MAE = ${metrics_gr.meanAbsoluteError}")
println(s"Explained variance = ${metrics_gr.explainedVariance}")

// COMMAND ----------

display(prediction_gr.select("Predicted appl_log", "appl_log", "volume_log","price_change", "volume_change"))

// COMMAND ----------

cvModel.extractParamMap

// COMMAND ----------

cvModel.explainParams

// COMMAND ----------

val glr = new GeneralizedLinearRegression()
   .setFamily("gaussian")
  .setLink("identity")
  .setMaxIter(10)
  .setRegParam(0.3)
  .setLabelCol(labelColumn)
  .setFeaturesCol("features")
  .setPredictionCol("Predicted " + labelColumn)

val pipeline_glr = new Pipeline()
  .setStages(Array(assembler, glr))

val model_glr = pipeline_glr.fit(trainingData)
//We'll make predictions using the model and the test data
val predictions_glr = model_glr.transform(testData)
//This will evaluate the error/deviation of the regression using the Root Mean Squared deviation
val evaluator_glr = new RegressionEvaluator()
            .setLabelCol(labelColumn)
            .setPredictionCol("Predicted " + labelColumn)
            .setMetricName("mse")
//We compute the error using the evaluator
val error = evaluator_glr.evaluate(predictions_glr)
println(error)

// COMMAND ----------

val out = predictions_glr
  .select("Predicted appl_log", "appl_log")
  .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
val metrics = new RegressionMetrics(out)
println(s"MSE = ${metrics.meanSquaredError}")
println(s"RMSE = ${metrics.rootMeanSquaredError}")
println(s"R-squared = ${metrics.r2}")
println(s"MAE = ${metrics.meanAbsoluteError}")
println(s"Explained variance = ${metrics.explainedVariance}")

// COMMAND ----------

glr.explainParams()

// COMMAND ----------

// Summarize the model over the training set and print out some metrics
    val summary = model.summary
    println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
    println(s"T Values: ${summary.tValues.mkString(",")}")
    println(s"P Values: ${summary.pValues.mkString(",")}")
    println(s"Dispersion: ${summary.dispersion}")
    println(s"Null Deviance: ${summary.nullDeviance}")
    println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
    println(s"Deviance: ${summary.deviance}")
    println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
    println(s"AIC: ${summary.aic}")
    println("Deviance Residuals: ")
    summary.residuals().show()
    // $example off$


// COMMAND ----------

//from pyspark.sql.functions import col  # for indicating a column using a string in the line below
//df = df.select([col(c).cast("double").alias(c) for c in df.columns])

// COMMAND ----------


