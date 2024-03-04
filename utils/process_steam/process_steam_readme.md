数据来源：https://steam.internet.byu.edu/ 

存档：
本地存储：D:\task\dataset\steam.sql.gz
网盘存储：阿里云盘/数据集/steam.sql.gz


关键指令：
1. 查看split_steam.1.sql文件第41行，前100个字符 sed -n '41,42p' split_steam.1.sql | cut -c -100

20221218
1. 抽取用户id，游戏的appid数据，以及各种auxiliary data；数据处理脚本存储在steam-v1文件夹下
2. 原始steam数据为sql格式，且原始文件极大，有160G，为方便处理，做了一些切分，划分为多个最大为9G的小文件
3. 划分后的数据中，从split_steam_12.sql开始，都是介绍player_summaries内容的，即用户相关的个人信息汇总