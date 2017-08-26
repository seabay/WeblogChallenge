package com.paytmlabs.challenge;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class BasicProcessOld implements Serializable {
	
	
	private static final long serialVersionUID = 5499001224848391232L;
	
	public final String sessionInfo = "data/session.log";

	public void process() {

		String path = "data/2015_07_22_mktplace_shop_web_log_sample.log";
		// String path = "data/sample.log";
		SparkSession spark = SparkSession.builder().master("local").appName("JavaPaytmlabsChallenge").config("spark.driver.host", "localhost").getOrCreate();

		DataFrameReader reader = spark.read();
		reader.option("sep", " ");
		reader.option("quote ", "\"");
		reader.schema(Schema.getSchema());
		Dataset<Row> lines = reader.csv(path); //
		lines.sort("clientIP", "timestamp");
		lines.show(20);
		JavaRDD<Row> jlines = lines.toJavaRDD();
		JavaRDD<BriefRecord> records = jlines.map(new GenBriefRecord());

		Dataset<Row> rows = spark.createDataFrame(records, BriefRecord.class);
		Dataset<Row> sortedRows = rows.sort("ip", "time");
		sortedRows.show(20);
		sortedRows.printSchema();

		analyse(sortedRows, sessionInfo);

		sessionize(spark, sessionInfo);
		
		spark.stop();
	}

	public void add2Dict(Map<String, List<SessionRecord>> session, String ip, SessionRecord sr) {

		if (session.containsKey(ip)) {
			session.get(ip).add(sr);
		} else {

			List<SessionRecord> vals = new ArrayList<SessionRecord>();
			vals.add(sr);
			session.put(ip, vals);
		}
	}

	public void analyse(Dataset<Row> rows, String path) {

		List<Row> list = rows.collectAsList();

		String curIP = "";
		double startTime = 0;
		double endTime = 0;

		String strStartTime = "";
		String strEndTime = "";

		Map<String, List<SessionRecord>> session = new HashMap<String, List<SessionRecord>>();

		boolean initial = true;

		Set<String> uniqUrl = new HashSet<String>();

		for (Row r : list) {

			String ip = r.getString(1);
			String url = r.getString(3);
			int elapse = r.getInt(0);
			String timestamp = r.getString(2);

			if (ip.compareTo(curIP) == 0) {
				if ((elapse - startTime) <= 900) {
					endTime = elapse;
					strEndTime = timestamp;
					uniqUrl.add(url);
				} else {

					SessionRecord sr = new SessionRecord(ip, strStartTime, strEndTime, endTime - startTime,
							uniqUrl.size(), 0);
					add2Dict(session, ip, sr);

					uniqUrl.clear();
					uniqUrl.add(url);
					startTime = elapse;
					endTime = elapse;

					strStartTime = timestamp;
					strEndTime = timestamp;
				}
			} else {

				if (initial) {
					initial = false;
				} else {

					SessionRecord sr = new SessionRecord(curIP, strStartTime, strEndTime, endTime - startTime,
							uniqUrl.size(), 0);
					add2Dict(session, curIP, sr);
					uniqUrl.clear();
				}

				curIP = ip;
				uniqUrl.add(url);
				startTime = elapse;
				endTime = elapse;
				strStartTime = timestamp;
				strEndTime = timestamp;
			}
		}

		SessionRecord sr = new SessionRecord(curIP, strStartTime, strEndTime, endTime - startTime, uniqUrl.size(), 0);
		add2Dict(session, curIP, sr);
		uniqUrl.clear();

		try {
			FileWriter fw = new FileWriter(sessionInfo);

			for (String k : session.keySet()) {

				List<SessionRecord> srList = session.get(k);

				for (SessionRecord item : srList) {

					double elapse = item.getElapse();
					if (elapse == 0.0)
						elapse = 1;
					int urlCount = item.getUrlCount();
					String st = item.getStart();
					String ed = item.getEnd();

					fw.write(k + "\t" + elapse + "\t" + urlCount + "\t" + st + "\t" + ed + "\n");
				}
			}

			fw.close();

			session.clear();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void sessionize(SparkSession spark, String path) {

		DataFrameReader reader = spark.read();
		reader.option("sep", "\t");
		reader.schema(Schema.getSessionRecordSchema());
		Dataset<Row> lines = reader.csv(path);
		lines.show();
		lines.printSchema();

		lines.createOrReplaceTempView("sessions");
		spark.sql(" SELECT avg(elapse) from sessions").show();
		spark.sql(" SELECT ip, sum(urlCount) as unique_url_count from sessions group by ip").show();
		spark.sql(" SELECT ip, sum(elapse) as session_length from sessions group by ip order by session_length desc")
				.show();
	}

//	public static void main(String[] args) throws Exception {
//
//		/// to solve "error - Relative path in absolute URI"
//		System.setProperty("spark.sql.warehouse.dir", "file:///C:/eclipse/eclipse-workspace/main/java/spark-warehouse");
//
//		BasicProcessOld bp = new BasicProcessOld();
//		bp.process();
//
//	}
}
