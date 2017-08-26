package com.paytmlabs.challenge;

import java.io.Serializable;
import java.util.Comparator;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;

public class BriefRecord implements Serializable{
	
	
	private static final long serialVersionUID = 8566371088878249455L;
	
	String ip;
	String time;
	int elapse;
	String url;
	
	public BriefRecord() {
		
	}
	
	public BriefRecord(String ip, String time, int elapse, String url) {
		super();
		this.ip = ip;
		this.time = time;
		this.elapse = elapse;
		this.url = url;
	}
	
	
	public String getIp() {
		return ip;
	}
	public void setIp(String ip) {
		this.ip = ip;
	}
	public String getTime() {
		return time;
	}
	public void setTime(String time) {
		this.time = time;
	}
	public int getElapse() {
		return elapse;
	}
	public void setElapse(int elapse) {
		this.elapse = elapse;
	}
	public String getUrl() {
		return url;
	}
	public void setUrl(String url) {
		this.url = url;
	}
}

class BriefRecordCompare implements Comparator<BriefRecord>{

	@Override
	public int compare(BriefRecord br1, BriefRecord br2) {
		
		if(br1.getElapse() > br2.getElapse())
			return 1;
		
		if(br1.getElapse() == br2.getElapse())
			return 0;
		
		return -1;
	}
	
}


class GenBriefRecord implements Function<Row, BriefRecord> {

	private static final long serialVersionUID = 7156748726833373027L;

	@Override
	public BriefRecord call(Row row) throws Exception {
		
		BriefRecord r = new BriefRecord();
		
		String time = row.getTimestamp(0).toString().split(" ")[1].split("\\.")[0];
		r.setTime(time);			
		
		String[] tms = time.split(":");
		int itime = Integer.parseInt(tms[0]) * 3600 + Integer.parseInt(tms[1]) * 60 + Integer.parseInt(tms[2]);
		r.setElapse(itime);

		String ip = row.getString(2).split(":")[0];
		r.setIp(ip);
		
		String url = "";
		if(row.length() >= 10)
			url = row.getString(11).replaceAll("\"", "").split(" ")[1];
		r.setUrl(url);
		
		return r;
	}
	
}
