package com.paytmlabs.challenge.ml;

import java.io.IOException;

import org.apache.log4j.PropertyConfigurator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
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

public class LRExpectedRegression {

	private String path = "data/second_load";

	private LinearRegression lr;
	private Dataset<Row>[] splits = null;
	private final String MODEL_PATH = "model/lr_model";

	public boolean loadData(SparkSession spark) {

		StructType schema = new StructType(
				new StructField[] { new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("hour", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("minute", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("second", DataTypes.IntegerType, false, Metadata.empty()) });

		DataFrameReader reader = spark.read();
		reader.option("sep", " ");
		reader.option("quote ", "\"");
		reader.schema(schema);
		Dataset<Row> data = reader.csv(path);

		splits = data.randomSplit(new double[] { 0.7, 0.3 }, 42);

		if (data.count() > 0)
			return true;
		return false;
	}

	public Pipeline buildPipeline() {

		lr = new LinearRegression().setLabelCol("label").setFeaturesCol("mergeFeature");

		OneHotEncoder oheHour = new OneHotEncoder().setDropLast(false).setInputCol("hour").setOutputCol("oneHotHour");
		OneHotEncoder oheMinute = new OneHotEncoder().setDropLast(false).setInputCol("minute")
				.setOutputCol("oneHotMinute");
		OneHotEncoder oheSecond = new OneHotEncoder().setDropLast(false).setInputCol("second")
				.setOutputCol("oneHotSecond");
		String[] cols = { "oneHotHour", "oneHotMinute", "oneHotSecond" };
		VectorAssembler va = new VectorAssembler().setInputCols(cols).setOutputCol("mergeFeature");

		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { oheHour, oheMinute, oheSecond, va, lr });

		return pipeline;
	}

	public CrossValidatorModel train(Pipeline pipeline, Dataset<Row> data) {

		ParamMap[] paramGrid = new ParamGridBuilder().addGrid(lr.regParam(), new double[] { 0.5, 0.2, 0.1, 0.01 })
				.addGrid(lr.elasticNetParam(), new double[] { 0.7, 0.6, 0.4, 0.1}).build();

		CrossValidator cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator())
				.setEstimatorParamMaps(paramGrid).setNumFolds(5);

		// Run cross-validation, and choose the best set of parameters.
		CrossValidatorModel cvModel = cv.fit(data);

		PipelineModel plm = (PipelineModel) cvModel.bestModel();

		LinearRegressionModel lrm = (LinearRegressionModel) (plm.stages()[4]);
		try {
			lrm.write().overwrite().save(MODEL_PATH);
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Elatic:"+lrm.getElasticNetParam() + "\tReg;"+lrm.getRegParam());

		return cvModel;
	}

	public void evalute(CrossValidatorModel cvModel, Dataset<Row> data) {

		Dataset<Row> predictions = cvModel.transform(data);

		// predictions.select("prediction", "label", "oneHotHour", "oneHotMinute",
		// "oneHotSecond").show(100);

		// Select (prediction, true label) and compute test error
		RegressionEvaluator evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction")
				.setMetricName("rmse");
		double rmse = evaluator.evaluate(predictions);
		System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
	}

	public void process() {

		SparkSession spark = SparkSession.builder().master("local").appName("JavaPaytmlabsChallenge")
				.config("spark.driver.host", "localhost").getOrCreate();

		loadData(spark);

		Pipeline pipeline = buildPipeline();

		CrossValidatorModel cvModel = train(pipeline, this.splits[0]);

		evalute(cvModel, this.splits[1]);

		spark.stop();
	}

	public static void main(String[] args) throws Exception {

		/// to solve "error - Relative path in absolute URI"
		System.setProperty("spark.sql.warehouse.dir", "file:///C:/eclipse/eclipse-workspace");
		PropertyConfigurator.configure("src/log4j.properties");
		LRExpectedRegression lrer = new LRExpectedRegression();
		lrer.process();

	}
}
