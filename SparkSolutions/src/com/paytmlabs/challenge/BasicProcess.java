package com.paytmlabs.challenge;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;

import scala.Tuple2;

@SuppressWarnings("serial")
public final class BasicProcess implements Serializable {

	//private static final Pattern SPACE = Pattern.compile(" (?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)");
	public final String sessionInfo = "data/session.log";

	class ParseLine implements PairFunction<String, String, BriefRecord> {

		@Override
		public Tuple2<String, BriefRecord> call(String s) throws Exception {
			
			String[] cols = s.split(" ", 12);
			
			String time = cols[0].split("T")[1].split("\\.")[0];
			String[] tms = time.split(":");
			int itime = Integer.parseInt(tms[0]) * 3600 + Integer.parseInt(tms[1]) * 60 + Integer.parseInt(tms[2]);

			String ip = cols[2].split(":")[0];

			String remain = cols[11];
			String[] rcols = remain.split(" ", 3);

			String url = rcols[1];
			
			BriefRecord br = new BriefRecord(ip, time, itime, url);

			return new Tuple2<>(ip, br);
		}
	}

	class CreateComb implements Function<BriefRecord, List<BriefRecord>> {

		@Override
		public List<BriefRecord> call(BriefRecord br) throws Exception {
			
			List<BriefRecord> lst = new ArrayList<BriefRecord>();
			lst.add(br);
			return lst;
		}
		
	}
	
	
	class MergeValue implements Function2<List<BriefRecord>, BriefRecord, List<BriefRecord>>{

		@Override
		public List<BriefRecord> call(List<BriefRecord> lst1, BriefRecord br) throws Exception {
			
			lst1.add(br);
			return lst1;
		}
	}
	
	class MergeComb implements Function2<List<BriefRecord>, List<BriefRecord>, List<BriefRecord>>{

		@Override
		public List<BriefRecord> call(List<BriefRecord> lst1, List<BriefRecord> lst2) throws Exception {
			
			lst1.addAll(lst2);
			return lst1;
		}
		
	}
	
	class ComputeSessionDuration implements Function<List<BriefRecord>, List<SessionRecord> >{

		@Override
		public List<SessionRecord> call(List<BriefRecord> lst) throws Exception {
			
			int size = lst.size();
			
			Collections.sort(lst, new BriefRecordCompare());
			
			String curIP = "";
			double startTime = 0;
			double endTime = 0;

			String strStartTime = "";
			String strEndTime = "";
			
			Set<String> uniqUrl = new HashSet<String>();
			List<String> urls = new ArrayList<String>();
			
			List<SessionRecord> ret = new ArrayList<SessionRecord>();
			
			for(int i=0 ;i<size; i++) {
				
				BriefRecord br = lst.get(i);
				
				String ip = br.getIp();
				String url = br.getUrl();
				int elapse = br.getElapse();
				String timestamp = br.getTime();

				if (ip.compareTo(curIP) == 0) {
					if ((elapse - startTime) <= 900) {
						endTime = elapse;
						strEndTime = timestamp;
						uniqUrl.add(url);
						urls.add(url);
					} else {

						SessionRecord sr = new SessionRecord(ip, strStartTime, strEndTime, endTime - startTime,
								uniqUrl.size(), urls.size());
						ret.add(sr);

						uniqUrl.clear();
						uniqUrl.add(url);
						
						urls.clear();
						urls.add(url);
						
						startTime = elapse;
						endTime = elapse;

						strStartTime = timestamp;
						strEndTime = timestamp;
					}
				} else {

					if (i == 0) {
						
					} else {

						SessionRecord sr = new SessionRecord(curIP, strStartTime, strEndTime, endTime - startTime,
								uniqUrl.size(), urls.size());
						ret.add(sr);
						uniqUrl.clear();
						urls.clear();
					}

					curIP = ip;
					uniqUrl.add(url);
					urls.add(url);
					startTime = elapse;
					endTime = elapse;
					strStartTime = timestamp;
					strEndTime = timestamp;
				}
			}
						
			SessionRecord sr = new SessionRecord(curIP, strStartTime, strEndTime, endTime - startTime, uniqUrl.size(),  urls.size());
			ret.add(sr);
			uniqUrl.clear();
			urls.clear();
					
			return ret;
		}
	}
	
	class PairMerge implements Function<Tuple2<String,List<SessionRecord>>,List<SessionRecord>>{

		@Override
		public List<SessionRecord> call(Tuple2<String, List<SessionRecord>> t) throws Exception {
			
			return t._2;
		}
	}

	public void process() {

		String rawData = "data/2015_07_22_mktplace_shop_web_log_sample.log";
		//String path = "data/sample.log";
		SparkSession spark = SparkSession.builder().master("local").appName("JavaPaytmlabsChallenge").config("spark.driver.host", "localhost").getOrCreate();

		JavaRDD<String> lines = spark.read().textFile(rawData).javaRDD().cache();

		JavaPairRDD<String, BriefRecord> records = lines.mapToPair(new ParseLine());
		
		JavaPairRDD<String, List<BriefRecord>> rdComb = records.combineByKey(new CreateComb(), new MergeValue(), new MergeComb());
		
//		List<Tuple2<String, List<BriefRecord>>>  tmp = rdComb.take(1);
//		Tuple2<String, List<BriefRecord>> ttmp = tmp.get(0);
//		System.out.println(ttmp._1);
//		System.out.println("size:"+ttmp._2.size());
//		for(BriefRecord br: ttmp._2) {
//			System.out.println("----"+br);
//		}
		
		
		JavaPairRDD<String, List<SessionRecord>> ret = rdComb.mapValues(new ComputeSessionDuration());
		
		List<Tuple2<String, List<SessionRecord>>> tmp2 = ret.collect();
		
		FileWriter fw = null;
		try {
			fw = new FileWriter(sessionInfo);
			for(Tuple2<String, List<SessionRecord>> tup2: tmp2) {
				List<SessionRecord> tlist = tup2._2;
				for(SessionRecord sr: tlist) {
					fw.write(tup2._1 + "\t" + sr.getElapse() + "\t" + sr.getUniqueUrl() + "\t"+ sr.getUrlCount() + "\t" + sr.getStart() + "\t" + sr.getEnd() + "\n");
				}
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
		
		sessionize(spark, sessionInfo);
		
		spark.stop();
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
		
		spark.sql(" SELECT ip, sum(uniqueUrlCount) as unique_url_count, sum(totalUrlCount) as total_url_count from sessions group by ip").show();
		Dataset<Row> rows = spark.sql(" SELECT ip, sum(uniqueUrlCount) as unique_url_count from sessions group by ip");
		rows.write().mode(SaveMode.Overwrite).csv("data/ip_url");
		
		spark.sql(" SELECT ip, sum(elapse) as session_length from sessions group by ip order by session_length desc").show();
		rows = spark.sql(" SELECT ip, sum(elapse) as session_length from sessions group by ip order by session_length desc");
		rows.write().mode(SaveMode.Overwrite).csv("data/ip_total_session_length");
	}

	

	public static void main(String[] args) throws Exception {

		/// to solve "error - Relative path in absolute URI"
		System.setProperty("spark.sql.warehouse.dir", "file:///C:/eclipse/eclipse-workspace");

		BasicProcess bp = new BasicProcess();
		bp.process();

	}
}
