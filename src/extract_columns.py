# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 23:37:21 2017

@author: Administrator
"""


def parse():
    
    
    file1 = open("time.log", "w+")
    file2 = open("ip.log", "w+")
    
    tm_rds = {}
    ip_rds = {};
    for line in open("..\data\2015_07_22_mktplace_shop_web_log_sample.log", 'r'):
    #for line in open("sample.txt", 'r'):
        line = line.strip('\r').strip('\n');
        
        cols = line.split(" ");
        
        time = cols[0].split("T")[1].split(".")[0]
        raw_time = time;
        tms = time.split(":")
        time = int(tms[0])*3600 + int(tms[1])*60 + int(tms[2])
        time = str(time)
        ip = cols[2].split(":")[0];
        url = cols[12]
        if len(cols) > 12:
            tm_value = ip + "\t"+url+"\t"+time+"\n"
            
            if raw_time in tm_rds:
                tm_rds[raw_time].append(tm_value)
            else:
                tm_rds[raw_time] = [tm_value]
                
            ip_value = url+"\t"+time+"\n";
            ip_key = ip+"\t"+raw_time;
            if ip_key in ip_rds:
                ip_rds[ip_key].append(ip_value);
            else:
                ip_rds[ip_key] = [ip_value]
            
            
    #records = sorted(records.items(), key=lambda d:d[0])    

    for k in sorted(tm_rds.keys()):
        for item in tm_rds[k]:
            file1.write(k+"\t"+item)
            
    for k in sorted(ip_rds.keys()):
        for item in ip_rds[k]:
            file2.write(k+"\t"+item)
      
    file1.close();
    file2.close();
        
if __name__ == '__main__':
    parse()