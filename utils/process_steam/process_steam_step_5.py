<<<<<<< Updated upstream
from time import time, strptime
from collections import defaultdict
import re

user_play_apps_dict = defaultdict(list)
user_play_time_dict = defaultdict(list)
user_play_two_weeks_time_dict = defaultdict(list)
app_dict = defaultdict(list)
time_set = defaultdict(set)
two_weeks_times = []
forever_times = []

with open('/root/autodl-nas/ltv/steam_games_1_records_2nd_filter.csv', 'rb') as f:
    tmp_x = time()
    for line_idx, line in enumerate(f):
        line = str(line) 
        tmp = line.strip('\r\n').split(',')
        
        str = ''
        crawl_time = str.join(re.findall("[0-9]|:|-| ", tmp[4]))
        
        if '"' in tmp[0]:
            user_steam_id = tmp[0].split('"')[1]
        elif "'" in tmp[0]:
            user_steam_id = tmp[0].split("'")[1]
        app_id = tmp[1]

        #import pdb; pdb.set_trace()
        user_play_apps_dict[user_steam_id].append(app_id)
        user_play_time_dict[user_steam_id].append(int(tmp[3]))
        user_play_two_weeks_time_dict[user_steam_id].append(int(tmp[2]))
        #app_dict[app_id].add(user_steam_id)

        #two_weeks_times.append(int(tmp[2]))
        #forever_times.append(int(tmp[3]))
        
        #time_set.add(tmp[4])

        #import pdb; pdb.set_trace()

        if line_idx % 5000000 == 0:
            t1 = time()
            print('Cost: %.4fs, Current has read %d lines in current file' % (t1 - tmp_x, line_idx))
            tmp_x = t1
=======
# 对第二批处理后的数据进行分析，即 steam_games_1_records_2nd_filter.csv文件，分析用户数量，app数量，code类似steam_v3_data.py
# app数量为 2,608，用户数量为 7,585,401 总记录为 17,353,006 
# 数据存在两个问题，用户太多了，app数量很少；传统的协同过滤算法还可以使用吗？统计的时间信息可能是不准确的，比如有些游戏用户只添加了少数时间
# 评估手段：用户历史花费时间预测

# 需要统计的其他内容：单用户在多个app上花销时间；单app上不同用户的花销时间 
# 同一个用户在不同 app 上花销时间的统计时间

# 可以根据 app 的用户个数，以及不同用户的花销时间，绘制一些图片，比如密度图
# 也可以绘制同一个用户在不同 app 上的花销时间

from time import time, strptime
from collections import defaultdict

user_dict = defaultdict(set)
app_dict = defaultdict(set)
time_set = defaultdict(set)
two_weeks_times = []
forever_times = []

with open('D:/task/dataset/steam-v1/steam_games_1_records_2nd_filter.csv', 'rb') as f:
    tmp_x = time()
    for line_idx, line in enumerate(f):
        line = str(line) 
        tmp = line.strip('\r\n').split(',')

        if '"' in tmp[0]:
            user_steam_id = tmp[0].split('"')[1]
        elif "'" in tmp[0]:
            user_steam_id = tmp[0].split("'")[1]
        app_id = tmp[1]

        #import pdb; pdb.set_trace()
        user_dict[user_steam_id].add(app_id)
        app_dict[app_id].add(user_steam_id)

        two_weeks_times.append(int(tmp[2]))
        forever_times.append(int(tmp[3]))
        
        #time_set.add(tmp[4])

        #import pdb; pdb.set_trace()

        if line_idx % 5000000 == 0:
            t1 = time()
            print('Cost: %.4fs, Current has read %d lines in current file' % (t1 - tmp_x, line_idx))
            tmp_x = t1
import pdb; pdb.set_trace()
>>>>>>> Stashed changes
