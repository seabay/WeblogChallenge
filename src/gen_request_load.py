# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:06:04 2017

@author: Administrator

Compute request counts per minute and per second

"""
import math

def process():
    
    start_time1 = 0;
    start_time2 = 0;
    
    
    init1 = True;
    init2 = True
    
    count1 = 0;
    count2 = 0;
    
    result1 = {}
    result2 = {}
    
    file1 = open("predict_load\minute_load.data", 'w')
    file2 = open("predict_load\second_load.data", 'w')
    
    for line in open("time.log", 'r'):
        
        line = line.strip('\r').strip('\n');
        cols = line.split("\t");
        
        time = cols[0].split(":");
        hour = time[0]
        minu = time[1]

        minu_key = hour+":"+minu;
        sec_key =  cols[0]      
        
        
        if init1:
            start_time1 = minu_key;
            count1 = 1;
            init1 = False;
        else:
            if minu_key == start_time1:
                count1 += 1;
            else:
                result1[start_time1] = count1;
                start_time1 = minu_key;
                count1 = 1;
                
        if init2:
            start_time2 = sec_key;
            count2 = 1;
            init2 = False;
        else:
            if sec_key == start_time2:
                count2 += 1;
            else:
                result2[start_time2] = count2;
                start_time2 = sec_key;
                count2 = 1;
                
                
        
    for k in sorted(result1.keys()):
        val = result1[k]
        if val>=500:
            val = 500;
        cat = math.ceil(val/50);
        file1.write(k + "\t" + str(result1[k]) + "\t"+ str(cat) + "\n")
        
    for k in sorted(result2.keys()):
        val = result2[k]
        if val>=500:
            val = 500;
        cat = math.ceil(val/50);
        file2.write(k + "\t" + str(result2[k]) + "\t"+ str(cat) + "\n")
        
    file1.close();
    file2.close();
    
if __name__ == '__main__':
    process()
            
                