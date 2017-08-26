package com.paytmlabs.challenge;

import java.io.Serializable;

import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Schema implements Serializable {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 3448805581080777770L;

	public static StructType getSchema()
	{
		StructType schema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("timestamp", DataTypes.TimestampType, true),
						DataTypes.createStructField("elb", DataTypes.StringType, false),
						DataTypes.createStructField("clientIP", DataTypes.StringType, false),
						DataTypes.createStructField("backendIP", DataTypes.StringType, true),
						DataTypes.createStructField("requestProcessingTime", DataTypes.DoubleType, true),
						DataTypes.createStructField("backendProcessingTime", DataTypes.DoubleType, true),
						DataTypes.createStructField("responseProcessingTime", DataTypes.DoubleType, true),
						DataTypes.createStructField("elbStatusCode", DataTypes.IntegerType, true),
						DataTypes.createStructField("backendStatusCode", DataTypes.IntegerType, true),
						DataTypes.createStructField("receivedBytes", DataTypes.DoubleType, true),
						DataTypes.createStructField("sentBytes", DataTypes.DoubleType, true),
						DataTypes.createStructField("request", DataTypes.StringType, true),
						DataTypes.createStructField("userAgent", DataTypes.StringType, true),
						DataTypes.createStructField("sslCipher", DataTypes.StringType, true),
						DataTypes.createStructField("sslProtocol", DataTypes.StringType, true) });
		
		
		return schema;
	}
	
	public static StructType getSessionRecordSchema()
	{
		StructType schema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("ip", DataTypes.StringType, false),
						DataTypes.createStructField("elapse", DataTypes.DoubleType, true),
						DataTypes.createStructField("uniqueUrlCount", DataTypes.IntegerType, true),
						DataTypes.createStructField("totalUrlCount", DataTypes.IntegerType, true),
						DataTypes.createStructField("startTime", DataTypes.StringType, true),
						DataTypes.createStructField("endTime", DataTypes.StringType, true) });
		
		
		return schema;
	}
}
