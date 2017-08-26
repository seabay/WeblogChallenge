package com.paytmlabs.challenge.ml;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;


public final class GenTraningData implements Serializable{

	private static final long serialVersionUID = -8437837687125585194L;
	
	public final String sessionInfo = "data/session.log";
	
	
	class ParseLine implements PairFunction<String, String, Integer> {

		private static final long serialVersionUID = 6843488844271902864L;

		@Override
		public Tuple2<String, Integer> call(String s) throws Exception {
			
			String[] cols = s.split(" ", 12);
			
			String time = cols[0].split("T")[1].split("\\.")[0];
			
			String[] tms = time.split(":");
		
			//Time t = new Time(tms[0], tms[1], tms[2]);
			
			//String t = " 1:"+ Integer.parseInt(tms[0]) + " 2:" + Integer.parseInt(tms[1]) + " 3:" + Integer.parseInt(tms[2]);
			String t = " "+ Integer.parseInt(tms[0]) + " " + Integer.parseInt(tms[1]) + " " + Integer.parseInt(tms[2]);

			return new Tuple2<>(t, 1);
		}
	}
	
	class ParseLine2 implements Function<String, Time> {

		private static final long serialVersionUID = 7019362250201931967L;

		@Override
		public Time call(String s) throws Exception {
			
			String[] cols = s.split(" ", 12);
			
			String time = cols[0].split("T")[1].split("\\.")[0];
			
			String[] tms = time.split(":");
		
			Time t = new Time(tms[0], tms[1], tms[2]);
			
			//String t = tms[0] + "\t" + tms[1] + "\t" + tms[2] + "\t1";

			return t;
		}
	}
	
	class Reducer implements Function2<Integer, Integer, Integer>{

		private static final long serialVersionUID = 7902293392888948787L;

		@Override
		public Integer call(Integer v1, Integer v2) throws Exception {
			return v1+v2;
		}
	}
	
	
	public void process() {

		String rawData = "data/2015_07_22_mktplace_shop_web_log_sample.log";
		SparkSession spark = SparkSession.builder().appName("JavaPaytmlabsChallenge").getOrCreate();

		JavaRDD<String> lines = spark.read().textFile(rawData).javaRDD().cache();
		
		JavaPairRDD<String, Integer> timeCount = lines.mapToPair(new ParseLine());
		JavaPairRDD<String, Integer> timeCountReduce = timeCount.reduceByKey(new Reducer());
		List<Tuple2<String, Integer>> timeCountMerge = timeCountReduce.collect();
		
		FileWriter fw = null;
		try {
			fw = new FileWriter("data/second_load");
			
			for(Tuple2<String, Integer> t:timeCountMerge) {
				
				fw.write(t._2 + t._1+"\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				fw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		spark.stop();
	}
	
	public void test() {
		
		String rawData = "data/2015_07_22_mktplace_shop_web_log_sample.log";
		SparkSession spark = SparkSession.builder().appName("JavaPaytmlabsChallenge").getOrCreate();

		JavaRDD<String> lines = spark.read().textFile(rawData).javaRDD().cache();
		
		JavaRDD<Time> timeUrl = lines.map(new ParseLine2());
		Dataset<Row> rows = spark.createDataFrame(timeUrl, Time.class);
		
		rows.cache();
		
		rows.show();
		rows.printSchema();

		rows.createOrReplaceTempView("sessions");
		
		spark.sql(" SELECT hour, minute, second, sum(urlCount) from sessions group by hour, minute, second order by hour, minute, second").show();
		
	}
	
	public static void main(String[] args) throws Exception {

		/// to solve "error - Relative path in absolute URI"
		System.setProperty("spark.sql.warehouse.dir", "file:///C:/eclipse/eclipse-workspace");

		GenTraningData gtd = new GenTraningData();
		gtd.process();

	}
}
