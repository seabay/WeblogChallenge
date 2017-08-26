package com.paytmlabs.challenge;

import java.io.Serializable;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class SessionRecord implements Serializable {

	private static final long serialVersionUID = -8245071124178936944L;	
	
	String ip;
	String start;
	String end;
	double elapse;
	int urlCount; /// total urls 
	int uniqueUrl; // unique urls

	public int getUniqueUrl() {
		return uniqueUrl;
	}

	public void setUniqueUrl(int uniqueUrl) {
		this.uniqueUrl = uniqueUrl;
	}

	public SessionRecord() {

	}

	public SessionRecord(String ip, String start, String end, double elapse, int uniqueUrl, int urlCount) {
		super();
		this.ip = ip;
		this.start = start;
		this.end = end;
		this.elapse = elapse;
		this.uniqueUrl = uniqueUrl;
		this.urlCount = urlCount;
	}

	public String getIp() {
		return ip;
	}

	public void setIp(String ip) {
		this.ip = ip;
	}

	public double getElapse() {
		return elapse;
	}

	public void setElapse(double elapse) {
		this.elapse = elapse;
	}

	public int getUrlCount() {
		return urlCount;
	}

	public void setUrlCount(int urlCount) {
		this.urlCount = urlCount;
	}

	public String getStart() {
		return start;
	}

	public void setStart(String start) {
		this.start = start;
	}

	public String getEnd() {
		return end;
	}

	public void setEnd(String end) {
		this.end = end;
	}

	@Override
	public String toString() {
		return "SessionRecord [ip=" + ip + ", start=" + start + ", end=" + end + ", elapse=" + elapse + ", urlCount="
				+ urlCount + "]";
	}
	
	

}

@SuppressWarnings("serial")
class GenSessionRecord implements Function2<Record, Record, SessionRecord> {

	@Override
	public SessionRecord call(Record r1, Record r2) throws Exception {

		SessionRecord r = new SessionRecord();

		String ip1 = r1.clientIP;
		String ip2 = r1.clientIP;

		if (ip1.compareTo(ip2) == 0) {
			r.setIp(ip2);
			;
			r.setElapse(r2.getElapse() - r1.getElapse());
			r.setUrlCount(2);
			r.setStart(r1.getTimestamp());
			r.setEnd(r2.getTimestamp());
		}

		return r;
	}
}

class GenPair implements PairFunction<Record, String, Record> {

	
	private static final long serialVersionUID = 6104132123279619654L;

	@Override
	public Tuple2<String, Record> call(Record r) throws Exception {
		String key = r.getClientIP();
		return new Tuple2<>(key, r);
	}
}
