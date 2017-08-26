package com.paytmlabs.challenge.ml;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class GBMExpectedLoad {

	private String path = "data/second_load";

	private GBTRegressor gbtr;

	public Dataset<Row> loadData(SparkSession spark) {

		StructType schema = new StructType(
				new StructField[] { new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("hour", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("minute", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("second", DataTypes.IntegerType, false, Metadata.empty()) });

		// Dataset<Row> data = spark.read().load(path);
		DataFrameReader reader = spark.read();
		reader.option("sep", " ");
		reader.option("quote ", "\"");
		reader.schema(schema);
		Dataset<Row> data = reader.csv(path);

		// data.show();
		// data.printSchema();

		// Split the data into training and test sets (30% held out for testing)
		// Dataset<Row>[] splits = data.randomSplit(new double[] { 0.7, 0.3 });

		return data;
	}

	public Dataset<Row> transform(Dataset<Row> data) {

		OneHotEncoder oheHour = new OneHotEncoder().setDropLast(false).setInputCol("hour").setOutputCol("oneHotHour");
		Dataset<Row> ds1 = oheHour.transform(data);

		OneHotEncoder oheMinute = new OneHotEncoder().setDropLast(false).setInputCol("minute")
				.setOutputCol("oneHotMinute");
		Dataset<Row> ds2 = oheMinute.transform(ds1);

		OneHotEncoder oheSecond = new OneHotEncoder().setDropLast(false).setInputCol("second")
				.setOutputCol("oneHotSecond");
		Dataset<Row> ds3 = oheSecond.transform(ds2);

		ds3.show();
		ds3.printSchema();

		return ds3;
	}

	public Pipeline buildPipeline() {

		gbtr = new GBTRegressor().setLabelCol("label").setFeaturesCol("oneHotSecond");
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { gbtr });
		return pipeline;
	}

	public CrossValidatorModel train(Pipeline pipeline, Dataset<Row> data) {

		Dataset<Row>[] splits = data.randomSplit(new double[] { 0.7, 0.3 });

		ParamMap[] paramGrid = new ParamGridBuilder().addGrid(gbtr.maxIter(), new int[] { 50, 100, 150 })
				.addGrid(gbtr.maxDepth(), new int[] { 1, 2, 3})
				.build();

		CrossValidator cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator())
				.setEstimatorParamMaps(paramGrid).setNumFolds(5); 

		// Run cross-validation, and choose the best set of parameters.
		CrossValidatorModel cvModel = cv.fit(splits[0]);

		Dataset<Row> predictions = cvModel.transform(splits[1]);
		for (Row r : predictions.select("label", "prediction").collectAsList()) {
			System.out.println("label, prediction (" + r.get(0) + ", " + r.get(1) + ")");
		}

		// Select example rows to display.
		System.out.println("Prediction...................");
		predictions.select("prediction", "label", "oneHotHour", "oneHotMinute", "oneHotSecond").show(100);

		// Select (prediction, true label) and compute test error
		RegressionEvaluator evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction")
				.setMetricName("rmse");
		double rmse = evaluator.evaluate(predictions);
		System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

		return cvModel;
	}

	public void process() {

		SparkSession spark = SparkSession.builder().master("local").appName("JavaPaytmlabsChallenge").config("spark.driver.host", "localhost").getOrCreate();

		Dataset<Row> data = loadData(spark);
		data = transform(data);

		Pipeline pipeline = buildPipeline();

		CrossValidatorModel cvModel = train(pipeline, data);

		spark.stop();
	}

	public static void main(String[] args) throws Exception {

		/// to solve "error - Relative path in absolute URI"
		System.setProperty("spark.sql.warehouse.dir", "file:///C:/eclipse/eclipse-workspace");

		GBMExpectedLoad gbme = new GBMExpectedLoad();
		gbme.process();

	}
}
