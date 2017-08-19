# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:14:24 2017

@author: Administrator
"""

def process():
    
    file1 = open("s_train.data", 'w+')
    file2 = open("m_train.data", 'w+')
    
    result = []
    for line in open("second_load.data", 'r'):
        line = line.strip('\r').strip('\n');
        cols = line.split("\t");
        time = cols[0].split(":")
        hour = str(int(time[0]))
        minute = str(int(time[1]))
        second = str(int(time[2]))
        
        elaps = int(time[0]) * 3600 + int(time[1])*60 + int(time[2])
        line = hour + "\t" + minute +"\t" + second + "\t" + cols[1] + "\t" + cols[2] + "\t" + str(elaps)
        result.append(line)
        
    for line in result:
        file1.write(line+"\n")
    
    file1.close();
    
    result = []
    for line in open("minute_load.data", 'r'):
        line = line.strip('\r').strip('\n');
        cols = line.split("\t");
        time = cols[0].split(":")
        hour = str(int(time[0]))
        minute = str(int(time[1]))
        
        elaps = int(time[0]) * 60 + int(time[1])
        line = hour + "\t" + minute +"\t" + cols[1] + "\t" + cols[2] + "\t" + str(elaps)
        result.append(line)
        
    for line in result:
        file2.write(line+"\n")
    
    file2.close();
    
if __name__ == '__main__':
    
    process()