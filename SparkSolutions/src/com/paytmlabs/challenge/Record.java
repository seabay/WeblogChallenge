package com.paytmlabs.challenge;

import java.io.Serializable;
import java.util.regex.Pattern;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;

public class Record implements Serializable {

	private static final long serialVersionUID = 2505623212326091267L;

	String timestamp="";
	int elapse = 0;

	String elb = "";
	String clientIP = "";
	String backendIP = "";
	double requestProcessingTime = 0;
	double backendProcessingTime = 0;
	double responseProcessingTime = 0;    //6
	int elbStatusCode = 0;
	int backendStatusCode = 0;
	double receivedBytes = 0;
	double sentBytes = 0;
	String request = "";
	String userAgent = "";
	String sslCipher = "";
	String sslProtocol = "";

	public int getElapse() {
		return elapse;
	}

	public void setElapse(int elapse) {
		this.elapse = elapse;
	}

	public String getTimestamp() {
		return timestamp;
	}

	public void setTimestamp(String timestamp) {
		this.timestamp = timestamp;
	}

	public String getElb() {
		return elb;
	}

	public void setElb(String elb) {
		this.elb = elb;
	}

	public String getClientIP() {
		return clientIP;
	}

	public void setClientIP(String clientIP) {
		this.clientIP = clientIP;
	}

	public String getBackendIP() {
		return backendIP;
	}

	public void setBackendIP(String backendIP) {
		this.backendIP = backendIP;
	}

	public double getRequestProcessingTime() {
		return requestProcessingTime;
	}

	public void setRequestProcessingTime(double requestProcessingTime) {
		this.requestProcessingTime = requestProcessingTime;
	}

	public double getBackendProcessingTime() {
		return backendProcessingTime;
	}

	public void setBackendProcessingTime(double backendProcessingTime) {
		this.backendProcessingTime = backendProcessingTime;
	}

	public double getResponseProcessingTime() {
		return responseProcessingTime;
	}

	public void setResponseProcessingTime(double responseProcessingTime) {
		this.responseProcessingTime = responseProcessingTime;
	}

	public int getElbStatusCode() {
		return elbStatusCode;
	}

	public void setElbStatusCode(int elbStatusCode) {
		this.elbStatusCode = elbStatusCode;
	}

	public int getBackendStatusCode() {
		return backendStatusCode;
	}

	public void setBackendStatusCode(int backendStatusCode) {
		this.backendStatusCode = backendStatusCode;
	}

	public double getReceivedBytes() {
		return receivedBytes;
	}

	public void setReceivedBytes(double receivedBytes) {
		this.receivedBytes = receivedBytes;
	}

	public double getSentBytes() {
		return sentBytes;
	}

	public void setSentBytes(double sentBytes) {
		this.sentBytes = sentBytes;
	}

	public String getRequest() {
		return request;
	}

	public void setRequest(String request) {
		this.request = request;
	}

	public String getUserAgent() {
		return userAgent;
	}

	public void setUserAgent(String userAgent) {
		this.userAgent = userAgent;
	}

	public String getSslCipher() {
		return sslCipher;
	}

	public void setSslCipher(String sslCipher) {
		this.sslCipher = sslCipher;
	}

	public String getSslProtocol() {
		return sslProtocol;
	}

	public void setSslProtocol(String sslProtocol) {
		this.sslProtocol = sslProtocol;
	}

	public static long getSerialversionuid() {
		return serialVersionUID;
	}
}




class GenRecordFromRow implements Function<Row, Record>{

	private static final long serialVersionUID = -5596210878191252078L;

	@Override
	public Record call(Row row) throws Exception {
		
		Record r = new Record();
		
		String time = row.getTimestamp(0).toString().split(" ")[1].split("\\.")[0];
		r.setTimestamp(time);			
		
		String[] tms = time.split(":");
		int itime = Integer.parseInt(tms[0]) * 3600 + Integer.parseInt(tms[1]) * 60 + Integer.parseInt(tms[2]);
		r.setElapse(itime);

		String ip = row.getString(2).split(":")[0];
		r.setClientIP(ip);
		
		String url = "";
		if(row.length() >= 10)
			url = row.getString(11).replaceAll("\"", "").split(" ")[1];
		r.setRequest(url);
		
		r.setElb(row.getString(1));
		r.setBackendIP(row.getString(3));
		
//		try {
//			double t = Double.parseDouble(cols[4]);
//			r.setRequestProcessingTime(t);
//		} catch (NumberFormatException e)
//		{
//			System.out.println(cols[4]);
//			System.out.println(s);
//		}
//		
		r.setRequestProcessingTime(row.getDouble(4));
		r.setBackendProcessingTime(row.getDouble(5));
		r.setResponseProcessingTime(row.getDouble(6));
		
		r.setElbStatusCode(row.getInt(7));
		r.setBackendStatusCode(row.getInt(8));
		
		r.setReceivedBytes(row.getDouble(9));
		r.setSentBytes(row.getDouble(10));
		
		r.setUserAgent(row.getString(12));
		r.setSslCipher(row.getString(13));
		r.setSslProtocol(row.getString(14));
		
		return r;
	}
	
}

class GenRecord implements Function<String, Record>{

	private static final long serialVersionUID = 8350008698360106518L;
	private static final Pattern SPACE = Pattern.compile(" (?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)");
	
	@Override
	public Record call(String s) throws Exception {
		
		Record r = new Record();
		
		String[] cols = SPACE.split(s);
		String time = cols[0].split("T")[1].split("\\.")[0];
		r.setTimestamp(time);			
		
		String[] tms = time.split(":");
		int itime = Integer.parseInt(tms[0]) * 3600 + Integer.parseInt(tms[1]) * 60 + Integer.parseInt(tms[2]);
		r.setElapse(itime);

		String ip = cols[2].split(":")[0];
		r.setClientIP(ip);
		
		String url = "";
		if(cols.length >= 10)
			url = cols[11].replaceAll("\"", "").split(" ")[1];
		r.setRequest(url);
		
		r.setElb(cols[1]);
		r.setBackendIP(cols[3]);
		
		try {
			double t = Double.parseDouble(cols[4]);
			r.setRequestProcessingTime(t);
		} catch (NumberFormatException e)
		{
			System.out.println(cols[4]);
			System.out.println(s);
		}
		
		r.setBackendProcessingTime(Double.parseDouble(cols[5]));
		r.setResponseProcessingTime(Double.parseDouble(cols[6]));
		
		r.setElbStatusCode(Integer.parseInt(cols[7]));
		r.setBackendStatusCode(Integer.parseInt(cols[8]));
		
		r.setReceivedBytes(Double.parseDouble(cols[9]));
		r.setSentBytes(Double.parseDouble(cols[10]));
		
		r.setUserAgent(cols[12]);
		r.setSslCipher(cols[13]);
		r.setSslProtocol(cols[14]);
		
		return r;
	}
	
}
