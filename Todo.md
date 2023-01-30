Todo:

+ ~~diversity怎么从数据中表现~~
+ ~~改装process_data代码，体现diversity~~
+ ~~更新LibRank代码，换用diversity指标~~
+ ~~用启发式方法得到baseline结果~~
+ ~~benchmark跑一遍 看指标~~
+ ~~调整diversity指标量级ILAD ERR_IA~~
+ ~~类目统计（一共多少cate， 每个多少item）~~
+ ~~多指标选哪个？~~
+ 电商数据集实验？
+ data.valid只选n=10
+ ~~1 2 3阶段   diversity~~
+ ~~all_div选div指标最好对应的map ndcg（指标选择有误）~~
+ ~~alpha-ndcg加入rl模型~~
+ ~~插入hypernetwork module（网络一部分替换成hypernet的mlp）~~
+ controllable模型：
  + 改进：lr调小、prm前面再加一层mlp hypernet、hypernetwork加深、multihead改为hypernet
  + 存最后一个图：x轴weight
  + 辅助判断：half PRM loss曲线、predition score的分布（差距？）
  + 所有轮里0最高的div，0.5最高的auc，1最高的auc
  + 两个loss之间的比对关系
  + w当作普通特征作为输入
  + w先固定，后调整
+ ~~seq2slate: label-smoothing改进~~
+ ~~EGRerank baseline~~
+ ~~大模型复现~~
  + ~~-evaluator(init)~~
+ EGR PPO实现方法baseline有问题（代码细节）
+ evaluator使用方式有问题（代码细节）
+ 线上page-wise打分，baseline item-wise打分
+ CMR只用label+discount
+ MMR
+ 随机数种子
+ 实验
  + dataset：ad（不常见介绍、基本参数：item多少个...）
  + baseline？根据什么规则挑选supervise seqential rl
  + performance
  + claim：
    + 通过输入参数来controllable是否work   does it work?
    + 我们的controllable是否足够effective             


Problem：

+ ~~reward？？？~~
+ ~~diversity怎么体现（通过指标吗）~~
+ ~~为什么会不一样？[0,0,0,1],[0,0,1,0]~~
+ ~~两个损失大小的量级~~
+ ~~指标不涨？~~
+ ~~数据怎么整理~~
+ ~~监督work？~~
+ ~~evaluator用处？~~
+ ~~CMR训练方式？~~
+ evaluator每次加载进来参数不一样？

周报：

+ ILAD指标：https://arxiv.org/pdf/2112.07621v1.pdf
+ ERR_IA指标：https://doi.org/10.1145/3442381.3449831
+ benchmark讲解：（feature：sparse_feature and dense_feature, ranker(lambdaMART) and reranker）
+ diversity指标情况（类目统计）item_num/cate_num（matplotlib）
+ 实验？原始实验？，加入loss_diverisity后，实验？
+ 折线图：指标~epoch    wandb

[TOC]

# 周报

## benchmark：[Neural Re-ranking in Multi-stage Recommender Systems: A Review](https://github.com/LibRerank-Community/LibRerank)

[Ad](https://tianchi.aliyun.com/dataset/56)是阿里巴巴提供的一个淘宝展示广告点击率预估数据集。

![image-20221116111456560](C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116111456560.png)

![image-20221116112130215](C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116112130215.png)

item_num=349494; cate_num=5525; 

item_spar_fnum=5 (item_id, cate_id, cam_id, cust_id, brand); 

item_dens_fnum=1 (price)

preprocess_ad.py(根据用户浏览广告的时间戳，将每个用户的记录转化为排行榜。在五分钟内交互过的项目被分成一个列表): [uid, profile, itm_spar_ft, itm_dens_ft, labels]

run_init_ranker.py(使用lambdaMART处理数据n选10): [out_user, profiles, out_itm_spar, out_itm_dens, out_label, out_pos, list_lens]

run_reranker.py: 训练reranker(10选[1, 3, 5 10])

![image-20221117134212295](C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221117134212295.png)

## 单目标to多目标

### SL

在AUC基础上借鉴[End-to-end Learnable Diversity-aware News Recommendation](https://arxiv.org/pdf/2204.00539.pdf)和[Diversification-Aware Learning to Rank using Distributed Representation](https://doi.org/10.1145/3442381.3449831)思想

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116120922192.png" alt="image-20221116120922192" style="zoom:50%;" />



<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116121011498.png" alt="image-20221116121011498" style="zoom:50%;" />

$L_{diversity}=\sum_{i=2}^{N}\sum_{j=1}^{i-1}\hat{y_i}\hat{y_j}\frac{1}{e^{\frac{|cate_i-cate_j|}{t}}+1}$

### RL

baseline选择：[Seq2slate](https://arxiv.org/pdf/1810.02019.pdf)（一次选一个item，监督学习训练）与[EGR](https://arxiv.org/pdf/2003.11941.pdf)

改动：借鉴[MDP-DIV](hp://dx.doi.org/10.1145/3077136.3080775)，将每一步ERR_IA指标的提升值作为reward(label)进行实验

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221124153441124.png" alt="image-20221124153441124" style="zoom:50%;" />

### diversity指标

+ ILAD: https://arxiv.org/pdf/2112.07621v1.pdf

  $\text { Diversity }=\operatorname{avg}_{i, j \in List, i \neq j}\left(1-\boldsymbol{S}_{i j}\right)$

+ ERR_IA：https://doi.org/10.1145/3442381.3449831

  $\mathrm{ERR}-\mathrm{IA}=\sum_{i=1}^n \frac{1}{r_i} \frac{1}{2^{c_{i}+1}}$，$c_i$为$rank$在$i$之前与$item_i$相同类目的item数量
  
+ $\alpha$-ndcg

### 验证集数据分析

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221117123706461.png" alt="image-20221117123706461" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221117123720204.png" alt="image-20221117123720204" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221117123744410.png" alt="image-20221117123744410" style="zoom:22%;" />

### PRM实验

#### all_auc

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116151801591.png" alt="image-20221116151801591" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116151922361.png" alt="image-20221116151922361" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116151942862.png" alt="image-20221116151942862" style="zoom:25%;" />

#### half auc half div

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221124154747880.png" alt="image-20221124154747880" style="zoom:27%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221124154819460.png" alt="image-20221124154819460" style="zoom:27%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221124154856076.png" alt="image-20221124154856076" style="zoom:27%;" />

#### all_div

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116231939391.png" alt="image-20221116231939391" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116232000744.png" alt="image-20221116232000744" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116232017228.png" alt="image-20221116232017228" style="zoom:22%;" />

| PRM/AD(acc_prefer)    | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@3  | ilad@5  | err_ia@3 | err_ia@5 | err_ia@10 |
| --------------------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | --------- |
| init                  | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64620 | 0.64535 | 1.26906  | 1.29619  | 1.32672   |
| all_auc               | 0.60485 | 0.60860 | 0.68469 | 0.69922 | 0.64341 | 0.64231 | 1.26745  | 1.29435  | 1.32527   |
| 0.9(max_auc)          | 0.60368 | 0.60765 | 0.68372 | 0.69853 | 0.64095 | 0.64190 | 1.26719  | 1.29368  | 1.32490   |
| 0.6(max_auc)          | 0.60312 | 0.60657 | 0.68394 | 0.69779 | 0.64364 | 0.64459 | 1.26804  | 1.29521  | 1.32591   |
| 0.3(max_auc)          | 0.60291 | 0.60658 | 0.68356 | 0.69782 | 0.64184 | 0.64341 | 1.26758  | 1.29457  | 1.32547   |
| 0.1(max_auc)          | 0.60219 | 0.60572 | 0.68305 | 0.69723 | 0.64318 | 0.64408 | 1.26788  | 1.29494  | 1.32579   |
| all_div(i3+e3+e5+e10) | 0.59936 | 0.60295 | 0.68071 | 0.69511 | 0.64626 | 0.64533 | 1.26909  | 1.29620  | 1.32673   |

### miDNN实验

#### all_auc

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116153246983.png" alt="image-20221116153246983" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116153302152.png" alt="image-20221116153302152" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116153325804.png" alt="image-20221116153325804" style="zoom:22%;" />

#### half auc half div

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221123122958996.png" alt="image-20221123122958996" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221123123020665.png" alt="image-20221123123020665" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221123123037388.png" alt="image-20221123123037388" style="zoom:22%;" />

#### all_div

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116193850818.png" alt="image-20221116193850818" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116193901375.png" alt="image-20221116193901375" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221116193934243.png" alt="image-20221116193934243" style="zoom:22%;" />

| miDNN/AD        | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@3  | ilad@5  | err_ia@3 | err_ia@5 | err_ia@10 |
| --------------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | --------- |
| init            | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64620 | 0.64535 | 1.26906  | 1.29619  | 1.32672   |
| all_auc         | 0.60076 | 0.60446 | 0.68198 | 0.69616 | 0.64267 | 0.64306 | 1.26735  | 1.29409  | 1.32504   |
| half(max_auc@5) | 0.60073 | 0.60445 | 0.68197 | 0.69615 | 0.64197 | 0.64329 | 1.26712  | 1.29396  | 1.32490   |
| all_div(e5)     | 0.58793 | 0.59207 | 0.67153 | 0.68686 | 0.64493 | 0.64544 | 1.26853  | 1.29592  | 1.32645   |

### Seq2slate实验

#### all_auc

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208123539571.png" alt="image-20221208123539571" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208123617295.png" alt="image-20221208123617295" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208123645713.png" alt="image-20221208123645713" style="zoom:25%;" />

#### half

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221207121015111.png" alt="image-20221207121015111" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221207121032231.png" alt="image-20221207121032231" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221207121045928.png" alt="image-20221207121045928" style="zoom:25%;" />

#### all_div

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221207205654198.png" alt="image-20221207205654198" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221207205708813.png" alt="image-20221207205708813" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221207205732574.png" alt="image-20221207205732574" style="zoom:25%;" />

| Seq2slate/AD(acc_prefer) | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@3  | ilad@5  | err_ia@3 | err_ia@5 | err_ia@10 |
| ------------------------ | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | --------- |
| init                     | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64620 | 0.64535 | 1.26906  | 1.29619  | 1.32672   |
| all_auc                  | 0.60456 | 0.60828 | 0.68481 | 0.69910 | 0.64483 | 0.64470 | 1.26847  | 1.29556  | 1.32622   |
| half                     | 0.60106 | 0.60473 | 0.68216 | 0.69651 | 0.64391 | 0.64457 | 1.26824  | 1.29544  | 1.32611   |
| all_div                  | 0.58435 | 0.58834 | 0.66893 | 0.68412 | 0.64623 | 0.64566 | 1.26912  | 1.29639  | 1.32685   |

### EGR实验

#### all_auc

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221205133228155.png" alt="image-20221205133228155" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221205133244507.png" alt="image-20221205133244507" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221205133303846.png" alt="image-20221205133303846" style="zoom:22%;" />

#### half

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221207114830339.png" alt="image-20221207114830339" style="zoom:30%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221207114908918.png" alt="image-20221207114908918" style="zoom:30%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221207114935087.png" alt="image-20221207114935087" style="zoom:30%;" />

#### all_div

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221206152831915.png" alt="image-20221206152831915" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221206152906132.png" alt="image-20221206152906132" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221206152927109.png" alt="image-20221206152927109" style="zoom:25%;" />

| EGR/AD(acc_prefer) | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@3  | ilad@5  | err_ia@3 | err_ia@5 | err_ia@10 |
| ------------------ | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | --------- |
| init               | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64620 | 0.64535 | 1.26906  | 1.29619  | 1.32672   |
| all_auc            | 0.60131 | 0.60507 | 0.68239 | 0.69664 | 0.64200 | 0.64306 | 1.26713  | 1.29396  | 1.32489   |
| half               | 0.60120 | 0.60484 | 0.68223 | 0.69647 | 0.64246 | 0.64331 | 1.26734  | 1.29419  | 1.32507   |
| all_div            | 0.58744 | 0.59135 | 0.67145 | 0.68643 | 0.64578 | 0.64543 | 1.26899  | 1.29622  | 1.32671   |

## Controllable(Hypernet)

### PRM

last week:

hypernet替换最后两层mlp，hypernet input:[auc_prefer, div_prefer]，output: 一层mlp的w和b，y=xw+b

训练方法（参考[Controllable Pareto Multi-Task Learning](https://arxiv.org/pdf/2010.06313.pdf)）：

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221201135409233.png" alt="image-20221201135409233" style="zoom:50%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221218180804629.png" alt="image-20221218180804629" style="zoom:50%;" />

每个batch取一个随机的preference vector（ex. [0.3, 0.7]），与auc和diversity两个loss做点乘相加梯度下降

attention Q K V换为hypernet生成；学习率调低了一半：1e-4 -> 5e-5；attention前加一层mlp

#### PRM实验

替换最后两层mlp:

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208124646307.png" alt="image-20221208124646307" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208124700778.png" alt="image-20221208124700778" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208124712676.png" alt="image-20221208124712676" style="zoom:25%;" />

替换最后两层mlp+学习率减半+attention中的mlp改为hypernet  Q K V:

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208124731397.png" alt="image-20221208124731397" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208124748067.png" alt="image-20221208124748067" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208124802540.png" alt="image-20221208124802540" style="zoom:25%;" />

替换最后两层mlp+学习率减半+attention前加一层hypernet mlp:

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208161126883.png" alt="image-20221208161126883" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208161145929.png" alt="image-20221208161145929" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221208161200751.png" alt="image-20221208161200751" style="zoom:25%;" />

### Seq2Slate

最后三层mlp改为hyper mlp

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213125341522.png" alt="image-20221213125341522" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213125352822.png" alt="image-20221213125352822" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213125406470.png" alt="image-20221213125406470" style="zoom:25%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213125802242.png" alt="image-20221213125802242" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213125815803.png" alt="image-20221213125815803" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213125831500.png" alt="image-20221213125831500" style="zoom:25%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221221215152477.png" alt="image-20221221215152477" style="zoom:25%;" />

| Seq2Slate/AD(auc_prefer) | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@5  | err_ia@5 | err_ia@10 |
| ------------------------ | ------- | ------- | ------- | ------- | ------- | -------- | --------- |
| init                     | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64535 | 1.29619  | 1.32672   |
| 0                        | 0.59085 | 0.59479 | 0.67406 | 0.68901 | 0.64528 | 1.29592  | 1.32647   |
| 0.5                      | 0.60098 | 0.60477 | 0.68193 | 0.69644 | 0.64500 | 1.2958   | 1.32641   |
| 1                        | 0.60142 | 0.60520 | 0.68241 | 0.69683 | 0.64481 | 1.29549  | 1.32615   |

### EGR_generator

最后三层mlp改为hyper mlp

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213124759653.png" alt="image-20221213124759653" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213124834870.png" alt="image-20221213124834870" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213124847101.png" alt="image-20221213124847101" style="zoom:25%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213124917125.png" alt="image-20221213124917125" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213124926719.png" alt="image-20221213124926719" style="zoom:25%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213125217144.png" alt="image-20221213125217144" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213125234752.png" alt="image-20221213125234752" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221213125310322.png" alt="image-20221213125310322" style="zoom:25%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221222110622000.png" alt="image-20221222110622000" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221222110637851.png" alt="image-20221222110637851" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221222110703841.png" alt="image-20221222110703841" style="zoom:22%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221222160757060.png" alt="image-20221222160757060" style="zoom:25%;" />

| EGR/AD(auc_prefer) | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@5  | err_ia@5 | err_ia@10 |
| ------------------ | ------- | ------- | ------- | ------- | ------- | -------- | --------- |
| init               | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64535 | 1.29619  | 1.32672   |
| 0                  | 0.58787 | 0.59207 | 0.67159 | 0.68686 | 0.64618 | 1.29652  | 1.32692   |
| 0.5                | 0.58859 | 0.59262 | 0.67215 | 0.68732 | 0.64632 | 1.29672  | 1.32705   |
| 1                  | 0.58936 | 0.59337 | 0.67278 | 0.68782 | 0.64570 | 1.29641  | 1.32694   |



## CMR

### Uncontrollable

| CMR/AD(acc_prefer) | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@3  | ilad@5  | err_ia@3 | err_ia@5 | err_ia@10 |
| ------------------ | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | --------- |
| init               | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64620 | 0.64535 | 1.26906  | 1.29619  | 1.32672   |
| all_auc            | 0.60041 | 0.60413 | 0.68170 | 0.69591 | 0.64180 | 0.64304 | 1.26711  | 1.29396  | 1.32489   |
| half               | 0.59628 | 0.60024 | 0.67882 | 0.69296 | 0.64518 | 0.64459 | 1.26839  | 1.29539  | 1.32596   |
| all_div            | 0.58810 | 0.59210 | 0.67176 | 0.68694 | 0.64591 | 0.64520 | 1.26895  | 1.29607  | 1.32664   |

### Controllable

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221221190834093.png" alt="image-20221221190834093" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221221190851890.png" alt="image-20221221190851890" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221221190916655.png" alt="image-20221221190916655" style="zoom:22%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221221212635641.png" alt="image-20221221212635641" style="zoom:17%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221221211932709.png" alt="image-20221221211932709" style="zoom:17%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221221211217417.png" alt="image-20221221211217417" style="zoom: 17%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221221224206553.png" alt="image-20221221224206553" style="zoom:17%;" />

| CMR/AD(auc_prefer) | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@5  | err_ia@5 | err_ia@10 |
| ------------------ | ------- | ------- | ------- | ------- | ------- | -------- | --------- |
| init               | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64535 | 1.29619  | 1.32672   |
| 0                  | 0.59252 | 0.59656 | 0.67543 | 0.69023 | 0.64495 | 1.29564  | 1.32620   |
| 0.5                | 0.59515 | 0.59919 | 0.67770 | 0.69221 | 0.64483 | 1.29572  | 1.32628   |
| 1                  | 0.59619 | 0.60029 | 0.67835 | 0.69301 | 0.64450 | 1.29517  | 1.32578   |

## MMR

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221224212327068.png" alt="image-20221224212327068" style="zoom:30%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20221224212253616.png" alt="image-20221224212253616" style="zoom:25%;" />

| MMR/AD(auc_prefer) | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@5  | err_ia@5 | err_ia@10 |
| ------------------ | ------- | ------- | ------- | ------- | ------- | -------- | --------- |
| init               | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64535 | 1.29619  | 1.32672   |
| 0                  | 0.59924 | 0.60270 | 0.68080 | 0.69511 | 0.66561 | 1.31469  | 1.34019   |
| 0.5                | 0.59961 | 0.60315 | 0.68096 | 0.69534 | 0.65651 | 1.30540  | 1.33350   |
| 1                  | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64535 | 1.29619  | 1.32672   |

## Robust Rank Aggregation

**主要方法：**

- Borda Count：对于每个ranking list，排名第一的文档获得n分，排名第二的文档获得n-1分，然后累加每个文档计算最终得分，按照最终得分给所有文档排序
- Markov Chain based ranking aggregation：利用不同的rank list统计文档转移概率，然后利用转移概率计算最佳的排序结果
- Cranking：利用不同的rank list结合马尔科夫蒙特卡洛搜索Markov Chain Monte Carlo (MCMC)

Borda Count实现Controllable：$score=score_{auc}*auc\_prefer+score_{div}*div\_prefer$

## CMR_generator

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230105112408345.png" alt="image-20230105112408345" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230105112431018.png" alt="image-20230105112431018" style="zoom:22%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230105112525088.png" alt="image-20230105112525088" style="zoom:22%;" />

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230105145116991.png" alt="image-20230105145116991" style="zoom:25%;" />

### fixed

| CMR/AD(auc_prefer) | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@5  | err_ia@5 | err_ia@10 |
| ------------------ | ------- | ------- | ------- | ------- | ------- | -------- | --------- |
| init               | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64535 | 1.29619  | 1.32672   |
| 0                  | 0.59279 | 0.59642 | 0.67552 | 0.69023 | 0.64757 | 1.29799  | 1.32803   |
| 0.5                |         |         |         |         |         |          |           |
| 1                  | 0.60147 | 0.60497 | 0.68246 | 0.69669 | 0.64399 | 1.29596  | 1.32677   |

### Controllable

| CMR/AD(auc_prefer) | map@5   | map@10  | ndcg@5  | ndcg@10 | ilad@5  | err_ia@5 | err_ia@10 |
| ------------------ | ------- | ------- | ------- | ------- | ------- | -------- | --------- |
| init               | 0.59961 | 0.60321 | 0.68092 | 0.69531 | 0.64535 | 1.29619  | 1.32672   |
| 0                  | 0.59579 | 0.59943 | 0.67803 | 0.69245 | 0.64569 | 1.29652  | 1.32698   |
| 0.5                | 0.59842 | 0.60202 | 0.67996 | 0.69437 | 0.64569 | 1.29643  | 1.32691   |
| 1                  | 0.60018 | 0.60381 | 0.68126 | 0.69574 | 0.64571 | 1.29643  | 1.32690   |

## CMR_evaluator











no_augmentation

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230111211953388.png" alt="image-20230111211953388" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230111210907020.png" alt="image-20230111210907020" style="zoom:25%;" />

with_augmentation

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230111214734822.png" alt="image-20230111214734822" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230111215708143.png" alt="image-20230111215708143" style="zoom:25%;" />

only_add_position_feature

<img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230112121813013.png" alt="image-20230112121813013" style="zoom:25%;" /><img src="C:\Users\chang\AppData\Roaming\Typora\typora-user-images\image-20230112121748075.png" alt="image-20230112121748075" style="zoom:25%;" />