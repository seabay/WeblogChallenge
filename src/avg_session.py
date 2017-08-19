# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:36:54 2017

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt


SESSION_TIME = 900 # 15*60

def sep_by_range(path):
    
    file = open(path, 'w+')
    
    cur_ip = '';
    start_time = 0;
    end_time = 0;
    
    session = {}
    
    initial = True;
    
    uniq_url = set();
    
    
    for line in open("ip.log", 'r'):
        line = line.strip('\r').strip('\n');
        cols = line.split("\t");
        
        ip = cols[0];
        url = cols[2];
        time = int(cols[3]);
        
        if ip == cur_ip:
            if (time-start_time) <= SESSION_TIME:
                end_time = time;
                uniq_url.add(url);
            else:
                
                if ip in session:
                    session[ip].append((end_time-start_time, len(uniq_url)))
                else:
                    session[ip] = [(end_time-start_time, len(uniq_url))]
                
                uniq_url.clear();
                uniq_url.add(url);
                start_time = time;
                end_time = time;
        else:
                    
            if initial:              
                initial = False;
            else:
                if cur_ip in session:
                    val = end_time-start_time
                    session[cur_ip].append((val, len(uniq_url)))
                else:
                    session[cur_ip] = [(end_time-start_time, len(uniq_url))]
                uniq_url.clear();
                    
            cur_ip = ip;    
            uniq_url.add(url);
            start_time = time;
            end_time = time;
            
    if cur_ip in session:
        val = end_time-start_time
        session[cur_ip].append((val, len(uniq_url)))
    else:
        session[cur_ip] = [(end_time-start_time, len(uniq_url))]
    uniq_url.clear();
           
    #file.write("ip\tsession\n")
    for k in sorted(session.keys()):
        for t in session[k]:           
            ses_len = t[0]
            url_cnt = t[1]
            if ses_len==0:
                ses_len=1
            file.write(k+"\t"+str(ses_len)+"\t"+str(url_cnt)+"\n")
    file.close();
    

def sep_by_gap(path):
    
    file = open(path, 'w+')
    
    cur_ip = '';
    start_time = 0;
    end_time = 0;
    
    session = {}
    
    initial = True;
    
    uniq_url = set();
    
    
    for line in open("ip.log", 'r'):
        line = line.strip('\r').strip('\n');
        cols = line.split("\t");
        
        ip = cols[0];
        url = cols[2];
        time = int(cols[3]);
        
        if ip == cur_ip:
            if (time-end_time) <= 600:
                
                end_time = time;
                
                uniq_url.add(url);
                
            else:
                
                if ip in session:
                    session[ip].append((end_time-start_time, len(uniq_url)))
                else:
                    session[ip] = [(end_time-start_time, len(uniq_url))]
                
                uniq_url.clear();
                uniq_url.add(url);
                start_time = time;
                end_time = time;
        else:
                    
            if initial:              
                initial = False;
            else:
                if cur_ip in session:
                    val = end_time-start_time
                    session[cur_ip].append((val, len(uniq_url)))
                else:
                    session[cur_ip] = [(end_time-start_time, len(uniq_url))]
                uniq_url.clear();
                    
            cur_ip = ip;    
            uniq_url.add(url);
            start_time = time;
            end_time = time;
            
    if cur_ip in session:
        val = end_time-start_time
        session[cur_ip].append((val, len(uniq_url)))
    else:
        session[cur_ip] = [(end_time-start_time, len(uniq_url))]
    uniq_url.clear();
           
    #file.write("ip\tsession\n")
    for k in sorted(session.keys()):
        for t in session[k]:           
            ses_len = t[0]
            url_cnt = t[1]
            if ses_len==0:
                ses_len=1
            file.write(k+"\t"+str(ses_len)+"\t"+str(url_cnt)+"\n")
    file.close();


def visual(path):
    
    data = pd.read_csv(path, sep='\t', names=['ip', 'session', 'url_count']);
    print(data.describe())
    
    #print(max(data['session']))
    
    data.hist(bins=100, column='session', figsize=(10,10))
    
    url_count = data['url_count']
    print(url_count.shape)
    #data.hist(bins=10000, column='url_count', figsize=(10,10))
    plt.show()
    
if __name__ == '__main__':
    #process()
    path1 = "predict_ip_info\ip_session.data";
    sep_by_range(path1)
    visual(path1)
    
    path2 = "predict_ip_info\ip_session_by_gap.data"
    sep_by_gap(path2)
    visual(path2)