package com.paytmlabs.challenge.ml;

import java.io.Serializable;

public class Time implements Serializable {

	private static final long serialVersionUID = 5743532523214159257L;
	
	String hour;
	String minute;
	String second;
	int urlCount = 1;
	
	public Time(String hour, String minute, String second) {
		super();
		this.hour = hour;
		this.minute = minute;
		this.second = second;
	}

	public String getHour() {
		return hour;
	}

	public void setHour(String hour) {
		this.hour = hour;
	}

	public String getMinute() {
		return minute;
	}

	public void setMinute(String minute) {
		this.minute = minute;
	}

	public String getSecond() {
		return second;
	}

	public void setSecond(String second) {
		this.second = second;
	}

	public int getUrlCount() {
		return urlCount;
	}

	public void setUrlCount(int urlCount) {
		this.urlCount = urlCount;
	}
	
	
	
}
