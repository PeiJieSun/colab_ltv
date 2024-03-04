# 统计过滤结束的数据，用户的数量有多少，app的数量有多少，统计时间有多少
# 原始数据文件大小：12.9G，用户个数：35901982，app个数：2703，其他数据均存在异常数值
# 先进行第二次筛选，保存文件名称为 steam_games_1_records_2nd_filter.csv，code文件见 steam_v4_data.py

from time import time

user_set = set()
app_set = set()
time_set = set()
two_weeks_times = []
forever_times = []

with open('D:/task/dataset/steam-v1/steam_games_1_records.csv', 'rb') as f:
    tmp_x = time()
    for line_idx, line in enumerate(f):
        line = str(line)
        tmp = line.strip('\r\n').split(',')
        #import pdb; pdb.set_trace()
        user_set.add(tmp[0])
        app_set.add(tmp[1])

        if tmp[2] != 'null':
            two_weeks_times.append(int(tmp[2]))
        if tmp[3] != 'null':
            forever_times.append(int(tmp[3]))
        
        time_set.add(tmp[4])

        if line_idx % 5000000 == 0:
            t1 = time()
            print('Cost: %.4fs, Current has read %d lines in current file' % (t1 - tmp_x, line_idx))
            tmp_x = t1
import pdb; pdb.set_trace()