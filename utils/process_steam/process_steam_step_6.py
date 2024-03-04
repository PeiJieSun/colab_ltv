# convert the steam data to atomic files

import time 
from collections import defaultdict
import re


steam_handle = open('D:\\task\dataset\steam-v1\steam_games_1.inter', 'w')
steam_handle.write('user_id:token\titem_id:token\ttwo_weeks_time:float\tcrawl_time:float\n')

#with open('/root/autodl-nas/ltv/steam_games_1_records_2nd_filter.csv', 'rb') as f:
with open('D:\\task\dataset\steam-v1\steam_games_1_records_2nd_filter.csv', 'rb') as f:
    tmp_x = time.time()
    for line_idx, line in enumerate(f):
        #print(line)
        line = str(line) 
        tmp = line.strip('\r\n').split(',')
        
        string = ''
        user_id = string.join(re.findall("[0-9]", tmp[0]))
        item_id = tmp[1]
        two_weeks_time = tmp[2]

        string = ''
        crawl_time = string.join(re.findall("[0-9]|:|-| ", tmp[4]))

        timeArray =  time.strptime(crawl_time, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(timeArray)

        steam_handle.write('%s\t%s\t%s\t%s\n' % (user_id, item_id, two_weeks_time, timestamp))
        
        if line_idx % 1000000 == 0:
            t1 =  time.time()
            print('Cost: %.4fs, Current has converted %d lines in current file' % (t1 - tmp_x, line_idx))
            tmp_x = t1