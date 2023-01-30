import pickle

import numpy as np
import pickle as pkl
from collections import defaultdict
import random
# from sklearn.metrics.pairwise import euclidean_distances
import argparse
import datetime
import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

test_dir = './Data/ad/lambdaMART.data.test'
test_lists = pkl.load(open(test_dir, 'rb'))

seq_len = test_lists[6]
cate_ids = list(map(lambda a: [i[1] for i in a], test_lists[2]))
cate_num = list(map(lambda a: len(set(a))-1 if 0 in set(a) else len(set(a)), cate_ids))

item_div_cate=[float(seq_len[i])/cate_num[i] for i in range(len(seq_len))]

plt.hist(seq_len)
# 显示横轴标签
plt.xlabel("")
# 显示纵轴标签
plt.ylabel("item_num")
# 显示图标题
plt.title("item_num per line in data.valid")
plt.show()

plt.hist(cate_num)
# 显示横轴标签
plt.xlabel("")
# 显示纵轴标签
plt.ylabel("cate_num")
# 显示图标题
plt.title("cate_num per line in data.valid")
plt.show()

plt.hist(item_div_cate)
# 显示横轴标签
plt.xlabel("")
# 显示纵轴标签
plt.ylabel("item/cate")
# 显示图标题
plt.title("item/cate per line in data.valid")
plt.show()

a = 1
