# 统计用户-app记录, steam_id, appid, playtime_2weeks, playtime_forever, dateretrieved
# 原始数据集中， playtime_2weeks, playtime_forever都有可能为null，接下来会检查 steam_id，appid，和dateretrieved 是否存在null
# 同时会将数据以 [steam_id, appid, playtime_2weeks, playtime_forever, dateretrieved] 的形式写到新的文件中

from time import time

steam_id_null_dict = {}
appid_id_null_dict = {}
dateretrieved_dict = {}

new_f = open('D:/task/dataset/steam-v1/steam_games_1_records.csv', 'w')

err_cnt = 0
with open('D:/task/dataset/steam-v1/steam_games_1.csv', 'rb') as f:
    tmp = time()

    for line_idx, line in enumerate(f):
        line = str(line)
        tmp1 = line.split('),(')

        for rec_idx, record in enumerate(tmp1):
            # 第一个记录需要特别处理
            if rec_idx == 0:
                record = record.split('(')[1]
                #import pdb; pdb.set_trace()
            # 其他记录均按照以下方式处理
            tmp2 = record.split(',')
            if len(tmp2) != 5:
                err_cnt += 1
            if len(tmp2) == 5:
                steam_id, appid, playtime_2weeks, playtime_forever, dateretrieved = tmp2[0], tmp2[1], tmp2[2], tmp2[3], tmp2[4]
            if steam_id == 'null' or appid == 'null' or 'dateretrieved' == 'null':
                import pdb; pdb.set_trace()
            if playtime_forever != 'null':
                new_line = ('%s,%s,%s,%s,%s\n' % (steam_id, appid, playtime_2weeks, playtime_forever, dateretrieved))
                new_f.write(new_line)
                #import pdb; pdb.set_trace()

        if line_idx % 100 == 0:
            t1 = time()
            print('Cost: %.4fs, Current has read %d lines in current file' % (t1 - tmp, line_idx))
            tmp = t1

print(err_cnt)
new_f.flush()
new_f.close()