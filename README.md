鲸鱼叫声互相关匹配法识别（正在更新中....）
每一个模块都可以单独运行，并生成对应的运行文件夹，包含运行日志、结果图、处理后的.pkl文件等。
运行顺序为：数据预处理模块（data_preprocessing.py）——>特征提取模块（feature_extraction.py）——>字典构建模块（correlation_dictionary.py）——>互相关匹配模块（cross_correlation_matching.py），互相关匹配模块运行时间较长。
