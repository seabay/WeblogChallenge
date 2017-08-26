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
