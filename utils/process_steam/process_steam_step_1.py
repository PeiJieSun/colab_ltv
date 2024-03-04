# 脚本作用：从原始steam数据集中抽取到用户在游戏app上过去两周的游戏时长

# 原始文件为steam.sql，为方便处理，已经按照最大文件大小分割
# 接下来将各个文件转换为csv格式

# 查看split_steam.1.sql文件第41行，前100个字符 sed -n '41p,42q' split_steam.1.sql | cut -c -100

# 数据集介绍 https://steam.internet.byu.edu/ 
# 处理数据集，主要搜集 用户过去两周在游戏上的花费时间。用于接下来的研究任务：根据用户在游戏上过去两周的花费时间，预测其在其他未观测游戏上的花费时间。
# 需要用到的数据包括 games_1, 可以获得user-steam-id, app id, playtime_2weeks, playtime_forever, dateretrieved 
# 还需要用户的历史游玩 app id，也存在一个新的问题，训练过程中可能会存在数据泄露问题，但是测试阶段不会出现数据泄露问题；容易出现一个问题就是过拟合现象，是否需要进行展开研究？
# 需要检查dateretrieved数据，确保构建的historical records是合理的
# 暂时只需要 games_1 数据

from time import time

data_dir = 'D:/task/dataset'

csv_name_list = ['achievement_percentages', 'app_id_info', 'friends', 'games_1', 'games_2', 'games_daily', 'games_developers', 'games_genres', 'games_publishers', 'groups', 'player_summaries']

'''
file_handle_list = []
for _, csv_name in enumerate(csv_name_list):
    f = open('D:/task/dataset/steam-v1/%s.csv' % csv_name, 'w')
    file_handle_list.append(f)
'''

write_f = open('D:/task/dataset/steam-v1/steam_%s.csv' % 'games_1', 'w')

# 实现进行文件内容查看，games_1数据可能只存在于 split_steam_4.sql, split_steam_5.sql, split_steam_6.sql, split_steam_7.sql 和 split_steam_8.sql
for file_idx in [3,4,5,6,7]:
    with open('D:/task/dataset/steam/split_steam.%d.sql' % (file_idx+1), 'rb') as f:
        t0 = time()
        tmp = t0
        for cnt_idx, line in enumerate(f):
            #import pdb; pdb.set_trace()
            line = str(line).lower()

            if 'games_1' in line:
                if 'INSERT INTO'.lower() in line:
                    #import pdb; pdb.set_trace()
                    write_f.write(line + '\n')
            '''    
            if 'INSERT INTO'.lower() in line:
                for csv_name_idx, csv_name in enumerate(csv_name_list):
                    if csv_name in line:
                        #import pdb; pdb.set_trace()
                        file_handle_list[csv_name_idx].write(line + '\n')
                        break
            '''
            if cnt_idx % 1000 == 0:
                t1 = time()
                print('Cost: %.4fs, Current has read %d lines in split_steam.%d.sql' % (t1 - tmp, cnt_idx, file_idx+1))
                tmp = t1
f.flush()
f.close()
        
        #for f in file_handle_list:
        #    f.flush()

'''
for f in file_handle_list:
    f.flush()
    f.close()
'''