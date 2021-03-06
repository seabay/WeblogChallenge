package com.paytmlabs.challenge.ml;

import java.io.IOException;
import java.io.Serializable;

import org.apache.log4j.PropertyConfigurator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
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

public final class RFExpectedLoad implements Serializable {

	private static final long serialVersionUID = -3937651597596935763L;
	private String path = "data/second_load";

	private RandomForestRegressor rf = null;
	private Dataset<Row>[] splits = null;
	
	private final String MODEL_PATH = "model/randomforest_model";

	public boolean loadData(SparkSession spark) {

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
		
		splits = data.randomSplit(new double[] { 0.7, 0.3 }, 42);

		if (data.count() > 0)
			return true;
		return false;
	}

	public Pipeline buildPipeline() {

		rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("mergeFeature");
		
		
		OneHotEncoder oheHour = new OneHotEncoder().setDropLast(false).setInputCol("hour").setOutputCol("oneHotHour");
		OneHotEncoder oheMinute = new OneHotEncoder().setDropLast(false).setInputCol("minute")
				.setOutputCol("oneHotMinute");
		OneHotEncoder oheSecond = new OneHotEncoder().setDropLast(false).setInputCol("second")
				.setOutputCol("oneHotSecond");
		String[] cols = {"oneHotHour", "oneHotMinute", "oneHotSecond"};
		VectorAssembler va = new VectorAssembler().setInputCols(cols).setOutputCol("mergeFeature");
		
		
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {oheHour,oheMinute,oheSecond, va, rf });
		return pipeline;
	}

	public CrossValidatorModel train(Pipeline pipeline, Dataset<Row> data) {

		ParamMap[] paramGrid = new ParamGridBuilder().addGrid(rf.numTrees(), new int[] { 30, 40, 50, 65, 80 })
				.addGrid(rf.maxDepth(), new int[] { 2, 4, 6})
				.build();

		CrossValidator cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator())
				.setEstimatorParamMaps(paramGrid).setNumFolds(5); 

		// Run cross-validation, and choose the best set of parameters.
		CrossValidatorModel cvModel = cv.fit(data);
		
		PipelineModel plm = (PipelineModel) cvModel.bestModel();
		
		RandomForestRegressionModel rfr = (RandomForestRegressionModel)(plm.stages()[4]);
		try {
			rfr.write().overwrite().save(MODEL_PATH);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println(rfr.toDebugString());
		System.out.println("NumTree:"+rfr.getNumTrees() + "\tMaxDepth:"+rfr.getMaxDepth());
		return cvModel;
	}
	
	
	public void evalute(CrossValidatorModel cvModel, Dataset<Row> data) {
		
		Dataset<Row> predictions = cvModel.transform(data);

		//predictions.select("prediction", "label", "oneHotHour", "oneHotMinute", "oneHotSecond").show(100);

		// Select (prediction, true label) and compute test error
		RegressionEvaluator evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction")
				.setMetricName("rmse");
		double rmse = evaluator.evaluate(predictions);
		System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
	}

	public void process() {

		SparkSession spark = SparkSession.builder().master("local").appName("JavaPaytmlabsChallenge").config("spark.driver.host", "localhost").getOrCreate();

		loadData(spark);
		
		Pipeline pipeline = buildPipeline();

		CrossValidatorModel cvModel = train(pipeline, this.splits[0]);

		evalute (cvModel, this.splits[1]);

		spark.stop();
	}

	public static void main(String[] args) throws Exception {

		/// to solve "error - Relative path in absolute URI"
		System.setProperty("spark.sql.warehouse.dir", "file:///C:/eclipse/eclipse-workspace");
		PropertyConfigurator.configure("src/log4j.properties");
		RFExpectedLoad rfe = new RFExpectedLoad();
		rfe.process();

	}

}
