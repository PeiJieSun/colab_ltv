# 由于第一次处理文件存在异常，现在进行第二次处理，移除two_weeks_times 为 null的数据，待处理文件名为 steam_games_1_records.csv，处理后文件名为 steam_games_1_records_2nd_filter.csv
# 处理完的数据集大小为1.03G，第二批数据的分析见steam_v5_data.py

from time import time

new_f = open('D:/task/dataset/steam-v1/steam_games_1_records_2nd_filter.csv', 'w')


with open('D:/task/dataset/steam-v1/steam_games_1_records.csv', 'rb') as f:
    tmp_x = time()
    for line_idx, line in enumerate(f):
        line = str(line)
        tmp = line.strip('\r\n').split(',')
        #import pdb; pdb.set_trace()
        if tmp[2] != 'null':
            new_f.write('%s\n' % line)

        if line_idx % 5000000 == 0:
            t1 = time()
            print('Cost: %.4fs, Current has read %d lines in current file' % (t1 - tmp_x, line_idx))
            tmp_x = t1

    f.close()
    new_f.close()