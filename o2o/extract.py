#分析top1中使用的特征，以及各个特征的相关性
#coupon相关feature，这部分最直观
import pandas as pd
import numpy as np

#
from datetime import date
def get_rate(x):
    #null代表没有折扣，即为1
    if x == "null":
        return 1
    
    #转换成折扣率
    if ":" in x:
        return 1- float(x.split(":")[1])/float(x.split(":")[0])
    else:
        return float(x)
 
def is_manjian(x):
    x = x.split(":")
    if len(x) == 1:
        return 0
    else:
        return 1
def get_man(x):
    #没有满减的如何赋值？应该是比较大的值，因为满越大，大家点的概率越小，这个需要验证
    #我认为这里赋值1为0不妥，因为这是成反比的，0会影响这个赋值
    #赋值为null，便于树方法，线性模型如何处理的？,最后都有一个null替换成nan
    if ":" not in  x:
        return "null"
    
    return int(x.split(":")[0])

def get_jian(x):
    #越小达标人家越不喜欢，赋值为0应该问题不大
    if ":" not in  x:
        return "null"
    
    return int(x.split(":")[1])

def get_dis(x):
    #
    if x == "null":
        return "null"
    
    return x

def get_weekday(x):
    #如果是null,onehot时都设为0
    if "null" == x:
        return "null"
    #20180401
    return date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1

def get_weektype(x):
    if x in [6,7]:
        return 1
    
def get_monthday(x):
    if "null" == x:
        return "null"
    
    return int(x[6:8])
    
    return 0
def coupon_feature(df):
    #这部分因为使用gbdt，去掉了一些onehot编码，看看效果如何，同时相对top缺少时间距离，但是这个我感觉问题不大，先试试


    #第一个就是折扣率，这个已经分析过，转换成折扣率，满额，减少额
    df["c_discount_rate"] = df.Discount_rate.apply(get_rate)
    
    #获取最大的满减，满值
    #df.Discount_rate
    #然后对于满减额，拆分成满和减，以及是否有满减
    df["c_is_manjian"] = df.Discount_rate.apply(is_manjian)
    df["c_man"] = df.Discount_rate.apply(get_man)
    #max_man = df.c_man.max()
    #df[df.c_man == 0]["c_man"] = max_man * 10 
    df["c_jian"] = df.Discount_rate.apply(get_jian)
    
    #然后就是距离，距离我认为由于有缺失值，onehot效果应该更好，待会可以验证
    
    #还有一个考虑到以后要用平均距离，最大距离这样的特征，貌似null替换成nan，然后mean操作可以忽略掉
    #df["distance"] = df.Distance.apply(get_dis)
    df["c_distance"] = df.Distance.apply(get_dis)
    #不保留null，对应lr模型有用
    #discols = ["c_dis_%s" %(i) for i in range(0,11)]
    #tmpdf = pd.get_dummies(df["distance"].replace("null", np.nan))
    #tmpdf.columns = discols
    #df[discols] = tmpdf   
    
    #时间特征，主要就是接收时间，是一周的第几天，一个月的第几天，是否是周末
    df["c_weekday"] = df.Date_received.astype('str').apply(get_weekday)
    df["c_weektype"] = df.c_weekday.apply(get_weektype)
    #onehot编码，对于lr模型比较有用
    #weekdaycols = ["c_weekday_%s" %(i) for i in range(1,8)]
    #tmpdf = pd.get_dummies(df["weekday"].replace("null", np.nan))
    #tmpdf.columns = weekdaycols
    #df[weekdaycols] = tmpdf
    #
    #d = df[['Coupon_id']]
    #d['c_coupon_count'] = 1
    #d = d.groupby('Coupon_id').agg('sum').reset_index()
    #coupon_count，每个coupon出现的数目
    #df = pd.merge(df,d,on='Coupon_id',how='left')    
    df["c_monthday"] = df.Date_received.astype('str').apply(get_monthday)
    #加上onehot
    #monthdaycols = ["c_monthday_%s" %(i) for i in range(1,32)]
    #tmpdf = pd.get_dummies(df["c_monthday"].replace("null",np.nan), prefix="c_monthday_")
    #df[tmpdf.columns] = tmpdf
    #replace null with nan
    #df.to_csv("./data.csv")
    
    return df

#开始预测，生辰标签
def get_label(x):
    #没有券的是-1
    if (x.Date_received == "null"):
        return -1
    #如果购买了
    if x.Date != 'null':
        #而且购买日期和领券日期相差不超过15天，值为1
        tmp = pd.to_datetime(x.Date, format="%Y%m%d") - pd.to_datetime(x.Date_received, format="%Y%m%d")
        #这个地方以前写成<了
        if tmp <= pd.Timedelta(15, 'D'):
            return 1    
    return 0
    #如果有券购买的算是1
#
#user realted
def user_feature(df):
    #用户相关的特征，考虑这么几个特征
    u = df[["User_id"]]
    u.drop_duplicates(inplace=True)
    
    #用户总的购买次数
    u1 = df[df.Date != 'null'][['User_id']]
    u1["u_buy_count"] = 1
    u1 = u1.groupby("User_id", as_index=False).count()
    
    #用户总的优惠券购买次数
    u2 = df[(df.Date != 'null') & (df.Date_received != 'null')][['User_id']]
    u2["u_coupon_buy_count"] = 1
    u2 = u2.groupby("User_id", as_index=False).agg('sum')
    #用户总的获取优惠券的次数
    u3 = df[(df.Date_received != 'null')][['User_id']]
    u3["u_coupon_count"] = 1
    u3 = u3.groupby("User_id", as_index=False).count()       
    #用户优惠券使用占比
    #用户优惠券购买占比
    
    #用户相关的折扣率，平均折扣率，这个总有值
    #这个作为特征反而降低了预测准确率，应该和改值的准确度有一定关系
    #u4 = df[(df.Date != 'null') & (df.Date_received != 'null')][['User_id', "c_discount_rate"]]
    #u4 = u4.groupby("User_id", as_index=False).agg('mean')
    #u4.rename(columns={"c_discount_rate":"u_mean_discount"}, inplace=True)
    
    
    #用户相关的距离,平均距离，最小，最大，平均，中位数
    tmpdf = df[(df.Date != 'null') & (df.Date_received != 'null')][['User_id', "Distance"]].copy()
    tmpdf.Distance.replace("null", -1, inplace=True)
    tmpdf.Distance = tmpdf.Distance.astype(int)
    tmpdf.replace(-1, np.nan, inplace=True)
    #
    u5 = tmpdf.groupby("User_id", as_index=False).agg('mean')
    u5.rename(columns = {'Distance':"u_mean_distance"}, inplace=True)
    #
    u6 = tmpdf.groupby("User_id", as_index=False).agg('max')
    u6.rename(columns = {'Distance':"u_max_distance"}, inplace=True)    
    u7= tmpdf.groupby("User_id", as_index=False).agg('min')
    u7.rename(columns = {'Distance':"u_min_distance"}, inplace=True)   
    u8= tmpdf.groupby("User_id", as_index=False).agg('median')
    u8.rename(columns = {'Distance':"u_median_distance"}, inplace=True)  
    
    #用户总的购买商户数，代表购买的不同商户个数
    u9 = df[df.Date != 'null'][['User_id', 'Merchant_id']]
    u9.drop_duplicates(inplace=True)
    u9["u_merchant_count"] = 1
    u9 = u9.groupby('User_id', as_index=False).agg(sum)
    u9 = u9[['User_id', 'u_merchant_count']]
    #用户从收券到消费的时间段
    u10 = df[(df.Date != 'null') & (df.Date_received !='null')][['User_id', 'Date_received', 'Date']]
    u10["date_datereceived"] = u10.Date_received.astype('str') + ":" + u10.Date.astype('str') 
    u10["date_gap"] = u10.date_datereceived.apply(get_gap)
    u10 = u10[["User_id", "date_gap"]]
    
    u11 = u10.groupby("User_id", as_index=False).mean()
    u11.rename(columns={"date_gap":"u_date_gap_mean"}, inplace=True)
    u12 = u10.groupby("User_id", as_index=False).min()
    u12.rename(columns={"date_gap":"u_date_gap_min"}, inplace=True)   
    u13 = u10.groupby("User_id", as_index=False).max()
    u13.rename(columns={"date_gap":"u_date_gap_max"}, inplace=True)     
    #用户相关的日期统计，是否是喜欢周末消费
    
    user_feature = pd.merge(u, u1, on=["User_id"], how='left')
    user_feature = pd.merge(user_feature, u2, on=["User_id"],how='left')
    user_feature = pd.merge(user_feature, u3, on=["User_id"], how='left')
    user_feature["u_coupon_buy_rate"] = user_feature.u_coupon_buy_count.astype(float)/user_feature.u_buy_count.astype(float)
    user_feature["u_coupon_use_rate"] = user_feature.u_coupon_buy_count.astype(float)/user_feature.u_coupon_count.astype(float)
    #user_feature = pd.merge(user_feature, u4, on=["User_id"], how='left')
    user_feature = pd.merge(user_feature, u5, on=["User_id"], how='left')
    user_feature = pd.merge(user_feature, u6, on=["User_id"], how='left')
    user_feature = pd.merge(user_feature, u7, on=["User_id"], how='left')
    user_feature = pd.merge(user_feature, u8, on=["User_id"], how='left')
    user_feature = pd.merge(user_feature, u9, on=["User_id"], how='left')
    user_feature = pd.merge(user_feature, u11, on=["User_id"], how='left')
    user_feature = pd.merge(user_feature, u12, on=["User_id"], how='left')
    user_feature = pd.merge(user_feature, u13, on=["User_id"], how='left')    
    #填充空值
    user_feature = user_feature.fillna(0)
    return user_feature
def get_gap(x):
    d1,d2 = x.split(":")
    return (date(int(d2[0:4]), int(d2[4:6]), int(d2[6:8])) - date(int(d1[0:4]), int(d1[4:6]), int(d1[6:8]))).days

def merchant_feature(df):
    #这部分多了date_gap相关特征，但是应该影响不大

    #用户相关的特征，考虑这么几个特征
    m = df[["Merchant_id"]]
    m.drop_duplicates(inplace=True)
    
    #merchant总的被购买次数
    m1 = df[df.Date != 'null'][['Merchant_id']]
    m1["m_buy_count"] = 1
    m1 = m1.groupby("Merchant_id", as_index=False).count()
    
    #merchant总的优惠券购买次数
    m2 = df[(df.Date != 'null') & (df.Date_received != 'null')][['Merchant_id']]
    m2["m_coupon_buy_count"] = 1
    m2 = m2.groupby("Merchant_id", as_index=False).count()    
    #用户总的获取优惠券的次数
    m3 = df[(df.Date_received != 'null')][['Merchant_id']]
    m3["m_coupon_count"] = 1
    m3 = m3.groupby("Merchant_id", as_index=False).count()       
    #用户优惠券使用占比
    #用户优惠券购买占比
    
    #用户相关的折扣率，平均折扣率，这个总有值
    #m4 = df[(df.Date != 'null') & (df.Date_received != 'null')][['Merchant_id', "c_discount_rate"]]
    #m4 = m4.groupby("Merchant_id", as_index=False).agg('mean')
    #m4.rename(columns={"c_discount_rate":"m_mean_discount"}, inplace=True)
    
    
    #用户相关的距离,平均距离，最小，最大，平均，中位数
    tmpdf = df[(df.Date != 'null') & (df.Date_received != 'null')][['Merchant_id', "Distance"]].copy()
    tmpdf.Distance.replace("null", -1, inplace=True)
    tmpdf.Distance = tmpdf.Distance.astype(int)
    tmpdf.replace(-1, np.nan, inplace=True)
    #
    m5 = tmpdf.groupby("Merchant_id", as_index=False).agg('mean')
    m5.rename(columns = {'Distance':"m_mean_distance"}, inplace=True)
    #
    m6 = tmpdf.groupby("Merchant_id", as_index=False).agg('max')
    m6.rename(columns = {'Distance':"m_max_distance"}, inplace=True)    
    m7= tmpdf.groupby("Merchant_id", as_index=False).agg('min')
    m7.rename(columns = {'Distance':"m_min_distance"}, inplace=True)   
    m8= tmpdf.groupby("Merchant_id", as_index=False).agg('median')
    m8.rename(columns = {'Distance':"m_median_distance"}, inplace=True)  
    
    #merchant总的coupon数
    #m9 = df[df.Coupon_id != 'null'][['Coupon_id', 'Merchant_id']]
    #m9.drop_duplicates(inplace=True);
    #m9["m_merchant_coupon_count"] = 1
    #m9 = m9.groupby('Merchant_id', as_index=False).agg(sum)
    #m9 = m9[['Merchant_id', 'm_merchant_coupon_count']]
    #用户从收券到消费的时间段
    m10 = df[(df.Date != 'null') & (df.Date_received !='null')][['Merchant_id', 'Date_received', 'Date']]
    m10["date_datereceived"] =  m10.Date_received + ":" +  m10.Date
    m10["date_gap"] = m10.date_datereceived.apply(get_gap)
    m10 = m10[["Merchant_id", "date_gap"]]
    
    m11 = m10.groupby("Merchant_id", as_index=False).mean()
    m11.rename(columns={"date_gap":"m_date_gap_mean"}, inplace=True)
    m12 = m10.groupby("Merchant_id", as_index=False).min()
    m12.rename(columns={"date_gap":"m_date_gap_min"}, inplace=True)   
    m13 = m10.groupby("Merchant_id", as_index=False).max()
    m13.rename(columns={"date_gap":"m_date_gap_max"}, inplace=True)     
    #用户相关的日期统计
    m_feature = pd.merge(m, m1, on=["Merchant_id"], how='left')
    m_feature = pd.merge(m_feature, m2, on=["Merchant_id"],how='left')
    m_feature = pd.merge(m_feature, m3, on=["Merchant_id"], how='left')
    m_feature["m_coupon_buy_rate"] = m_feature.m_coupon_buy_count.astype(float)/m_feature.m_buy_count.astype(float)
    m_feature["m_coupon_use_rate"] = m_feature.m_coupon_buy_count.astype(float)/m_feature.m_coupon_count.astype(float)
    #先不考虑平均折扣率
    #m_feature = pd.merge(m_feature, m4, on=["Merchant_id"], how='left')
    m_feature = pd.merge(m_feature, m5, on=["Merchant_id"], how='left')
    m_feature = pd.merge(m_feature, m6, on=["Merchant_id"], how='left')
    m_feature = pd.merge(m_feature, m7, on=["Merchant_id"], how='left')
    m_feature = pd.merge(m_feature, m8, on=["Merchant_id"], how='left')
    #m9和m3是重复的
    #m_feature = pd.merge(m_feature, m9, on=["Merchant_id"], how='left')
    #先参考top1
    m_feature = pd.merge(m_feature, m11, on=["Merchant_id"], how='left')
    m_feature = pd.merge(m_feature, m12, on=["Merchant_id"], how='left')
    m_feature = pd.merge(m_feature, m13, on=["Merchant_id"], how='left')    
    #填充空值
    m_feature = m_feature.fillna(0)
    return m_feature    
    #merchant的特征
    
    #merchant被购买的次数
    #merchant被领取券的次数
    #merchant被优惠券购买的次数
    #两个概率
    
    #merchant的折扣率
    
    #merchant的距离
    
    #merchant的被消费时间
    
def user_merchant(df):
    #用户和商品的交叉特征
    
    um = df[["User_id", "Merchant_id"]]
    um.drop_duplicates(inplace=True)
    
    #用户对某商品的购买次数
    um1 = df[df.Date !='null'][['User_id', 'Merchant_id']]
    um1["um_buy_count"] = 1
    um1 = um1.groupby(["User_id", "Merchant_id"], as_index=False).agg('sum')
    #这部分需要额外去重？groupby后不应该是不重复的
    um1.drop_duplicates(inplace=True)
    
    #用户使用优惠券对某商品的购买次数
    um2 = df[(df.Date !='null') & (df.Date_received != 'null')][['User_id', 'Merchant_id']]
    um2["um_buy_with_coupon_count"] = 1
    um2 = um2.groupby(["User_id", "Merchant_id"], as_index=False).agg('sum')
    um2.drop_duplicates(inplace=True)
    
    #用户总共收到的优惠券的数目
    um3 = df[df.Date_received !='null'][['User_id', 'Merchant_id']]
    um3["um_coupon_count"] = 1
    um3 = um3.groupby(["User_id", "Merchant_id"], as_index=False).agg('sum')
    um3.drop_duplicates(inplace=True)
 
    #交互次数
    um4 = df[['User_id', 'Merchant_id']]
    um4["um_count"] = 1
    um4 = um4.groupby(["User_id", "Merchant_id"], as_index=False).agg('sum')
    um4.drop_duplicates(inplace=True) 
    
    #common rate，消费了但没有使用优惠券
    um5 = df[(df.Date !='null') &(df.Coupon_id == 'null')][['User_id', 'Merchant_id']]
    um5["um_common_buy_count"] = 1
    um5 = um5.groupby(["User_id", "Merchant_id"], as_index=False).agg('sum')
    um5.drop_duplicates(inplace=True)
    
    #用户总共收到
    um_feature = pd.merge(um, um1, on=["User_id", 'Merchant_id'], how='left')
    um_feature = pd.merge(um_feature, um2, on=["User_id", 'Merchant_id'], how='left')
    um_feature = pd.merge(um_feature, um3, on=["User_id", 'Merchant_id'], how='left')
    um_feature = pd.merge(um_feature, um4, on=["User_id", 'Merchant_id'], how='left')
    um_feature = pd.merge(um_feature, um5, on=["User_id", 'Merchant_id'], how='left')
    um_feature.um_buy_count = um_feature.um_buy_count.replace(np.nan, 0)
    um_feature["um_buy_rate"] = um_feature.um_buy_count.astype('float')/um_feature.um_count.astype('float')
    um_feature.um_buy_with_coupon_count = um_feature.um_buy_with_coupon_count.replace(np.nan, 0)
    um_feature["um_buy_with_coupon_rate"] = um_feature.um_buy_with_coupon_count.astype('float')/um_feature.um_buy_count.astype('float')
    um_feature["um_coupon_transfer_rate"] = um_feature.um_buy_with_coupon_count.astype('float')/um_feature.um_coupon_count.astype('float')
    um_feature.um_common_buy_count = um_feature.um_common_buy_count.replace(np.nan, 0)
    um_feature["um_common_buy_rate"] = um_feature.um_common_buy_count.astype('float')/um_feature.um_buy_count.astype('float')
    um_feature = um_feature.fillna(0)
    
    return um_feature

#泄漏特征
def other_feature(test):
    #主要出发点就是test中如果同一用户收到同一优惠券次数越多，越表示这个券被消费了
    t = test[["User_id", "Coupon_id"]]
    t.drop_duplicates(inplace=True)
    
    #用户接受到总的券的数目，数目越大，被消费的可能性越大
    t1 = test[["User_id"]]
    t1["t_total_coupon_count"] = 1
    t1 = t1.groupby("User_id", as_index=False).agg('sum')
    
    #用户同一个券的接收次数
    t2 = test[["User_id", 'Coupon_id']]
    t2["t_same_coupon_count"] = 1
    t2 = t2.groupby(["User_id", "Coupon_id"], as_index=False).agg('sum')
    #单独构造是否大于1的特征
    t2["t_same_coupon_morethan1"]  = t2.t_same_coupon_count.apply(lambda x: 1 if x>1 else 0)
    
    #同一优惠券的最近接收日期和最小接收日期
    t3 = test[["User_id", "Coupon_id", "Date_received"]]
    t3.Date_received= t3.Date_received.astype('str')
    t3 = t3.groupby(['User_id','Coupon_id'])['Date_received'].agg(lambda x:':'.join(x)).reset_index()
    t3['receive_number'] = t3.Date_received.apply(lambda s:len(s.split(':')))
    t3 = t3[t3.receive_number>1]
    #中间变量
    t3["t_max_coupon_time"]= t3.Date_received.apply(lambda x:max([int(i) for i in x.split(":")]))
    t3["t_min_coupon_time"]= t3.Date_received.apply(lambda x:min([int(i) for i in x.split(":")]))
    t3 = t3[["User_id", "Coupon_id", "t_max_coupon_time", "t_min_coupon_time"]]
    #获取券和最小和最大接收时间的差值
    t4 = test[["User_id", "Coupon_id", "Date_received"]]
    t4 = pd.merge(t4, t3, on=["User_id", "Coupon_id"], how='left')
    t4["t_user_receive_same_coupon_lastone"] = t4.t_max_coupon_time-t4.Date_received.astype('int')
    t4["t_user_receive_same_coupon_firstone"] = t4.Date_received.astype('int') - t4.t_min_coupon_time
    #
    def is_firstlastone(x):
        if x==0:
            return 1
        elif x>0:
            return 0
        else:
            return -1 #those only receive once
    
    t4["t_user_receive_same_coupon_lastone"] = t4["t_user_receive_same_coupon_lastone"].apply(is_firstlastone)
    t4["t_user_receive_same_coupon_firstone"] = t4["t_user_receive_same_coupon_firstone"].apply(is_firstlastone)
    t4 = t4[["User_id", "Coupon_id",  "Date_received", "t_user_receive_same_coupon_lastone", "t_user_receive_same_coupon_firstone"]]
    #用户同一天收到的优惠券数目
    t5 = test[["User_id", "Date_received"]]
    t5["t_day_received_all_coupon_num"] = 1
    t5 = t5.groupby(['User_id', 'Date_received'], as_index=False).agg('sum')
    
    #同一天同一优惠券的数目
    t6 = test[["User_id", "Coupon_id", "Date_received"]]
    t6["t_day_received_same_coupon_num"] = 1
    t6 = t6.groupby(["User_id", "Coupon_id", "Date_received"], as_index=False).agg('sum')
    
    #
    t7 = test[['User_id', 'Coupon_id', 'Date_received']]
    t7.Date_received = t7.Date_received.astype('str')
    t7 = t7.groupby(['User_id','Coupon_id'])['Date_received'].agg(lambda x:':'.join(x)).reset_index()
    t7.rename(columns={'Date_received':'dates'},inplace=True)
    t8 = test[['User_id','Coupon_id','Date_received']]
    t8 = pd.merge(t8,t7,on=['User_id','Coupon_id'],how='left')
    t8['date_received_date'] = t8.Date_received.astype('str') + '-' + t8.dates
    t8['t_day_gap_before'] = t8.date_received_date.apply(get_day_gap_before)
    t8['t_day_gap_after'] = t8.date_received_date.apply(get_day_gap_after)
    t8 = t8[['User_id','Coupon_id','Date_received','t_day_gap_before','t_day_gap_after']]
    
    #
    other = pd.merge(t1, t2, on=["User_id"])
    other = pd.merge(other,t4, on=["User_id", "Coupon_id"])
    other = pd.merge(other,t5, on=["User_id", "Date_received"])
    other = pd.merge(other,t6, on=["User_id", "Coupon_id", "Date_received"])
    other = pd.merge(other,t8, on=["User_id", "Coupon_id", "Date_received"])
    #
    
    return other
    
def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1 #those only receive once
from datetime import date
def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    #对于每个消费日期，计算收到券的日期和消费时间的差值，如果消费后收券，这个间隔记做gap
    for d in dates:
        this_gap = (date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    #收券后消费的间隔的最小值
    for d in dates:
        this_gap = (date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)   
def usercouponFeature(df):

    dftmp = df[df.Coupon_id != 'null'].copy()
    
    
    uc = dftmp[['User_id', 'Coupon_id']].copy().drop_duplicates()

    #这是用户领取券的数目（包括消费的和未消费的）
    uc1 = dftmp[['User_id', 'Coupon_id']].copy()
    uc1['uc_count'] = 1
    uc1 = uc1.groupby(['User_id', 'Coupon_id'], as_index = False).count()

    #用户消费的券的数目
    uc2 = dftmp[dftmp['Date'] != 'null'][['User_id', 'Coupon_id']].copy()
    uc2['uc_coupon_buy_count'] = 1
    uc2 = uc2.groupby(['User_id', 'Coupon_id'], as_index = False).count()

    #接受了但是没消费的
    uc3 = dftmp[(dftmp['Date_received'] != 'null') & (dftmp["Date"] == "null")][['User_id', 'Coupon_id']].copy()
    uc3['uc_coupon_nosale_count'] = 1
    uc3 = uc3.groupby(['User_id', 'Coupon_id'], as_index = False).count()


    user_coupon_feature = pd.merge(uc, uc1, on = ['User_id','Coupon_id'], how = 'left')
    user_coupon_feature = pd.merge(user_coupon_feature, uc2, on = ['User_id','Coupon_id'], how = 'left')
    user_coupon_feature = pd.merge(user_coupon_feature, uc3, on = ['User_id','Coupon_id'], how = 'left')
    user_coupon_feature = user_coupon_feature.fillna(0)

    user_coupon_feature['uc_buy_rate'] = user_coupon_feature['uc_coupon_buy_count'].astype('float')/user_coupon_feature['uc_count'].astype('float')
    user_coupon_feature['uc_no_buy_rate'] = user_coupon_feature['uc_coupon_buy_count'].astype('float')/user_coupon_feature['uc_count'].astype('float')
    user_coupon_feature = user_coupon_feature.fillna(0)

    print(user_coupon_feature.columns.tolist())
    return user_coupon_feature    
def featureProcess(feature, train, test):
    """
    feature engineering from feature data
    then assign user, merchant, and user_merchant feature for train and test 
    """
    
    uf = user_feature(feature)
    mf = merchant_feature(feature)
    um = user_merchant(feature)
    #训练集合coupon和测试集合coupon没有交集，所以不适用user和coupon的交集了
    other = other_feature(test)
    #加上user copon
    #uc = usercouponFeature(feature)
    

    train = pd.merge(train, uf, on=["User_id"], how='left')
    train = pd.merge(train, mf, on=["Merchant_id"], how='left')
    train = pd.merge(train, um, on=[ "User_id", "Merchant_id"], how='left')
    train = pd.merge(train, other, on=["User_id", 'Coupon_id', "Date_received"], how='left')
    #train = pd.merge(train, uc, on=[ "User_id", "Coupon_id"], how='left')
    train = train.replace("null", np.nan)
    #
    train.drop_duplicates(inplace=True)
    train = train.fillna(0)
    #
    
    #test = pd.merge(test, uf, on = 'User_id', how = 'left')   
    #test = pd.merge(test, mf, on = 'Merchant_id', how = 'left')   
    #test = pd.merge(test, um, on=["User_id", "Merchant_id"], how='left')
    #test = pd.merge(test, uc, on=[ "User_id", "Coupon_id"], how='left')
    #test = pd.merge(test, other, on=["User_id", 'Coupon_id', "Date_received"], how='left')
    #test = test.replace("null", np.nan)
    #
    #test.drop_duplicates(inplace=True)
    #test = test.fillna(0)
    
    return train
#生成泄漏信息
def otherFeature(train, test):
    #
    other = other_feature(test)
    train = pd.merge(train, other, on=["User_id", 'Coupon_id', "Date_received"], how='left')
    train = train.fillna(0)
    test = pd.merge(test, other, on=["User_id", 'Coupon_id', "Date_received"], how='left')
    test = test.fillna(0)
    
    return train, test

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc
from sklearn.cross_validation import PredefinedSplit
import numpy as np
import pickle
def check_model(train, valid, predictors):
    #
    
    classifier = lambda: SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        max_iter=100, 
        shuffle=True, 
        n_jobs=1,
        class_weight=None)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])

    parameters = {
        'en__alpha': [ 0.001, 0.01, 0.1],
        'en__l1_ratio': [ 0.001, 0.01, 0.1]
    }
    #训练集是train+valid
    data = pd.concat([train, valid])
    #print(data[predictors].head())
    print("train size:%s, val size:%s, data size:%s" %(train.shape[0], valid.shape[0], data.shape[0]))
    #生成验证集
    index = np.zeros(data.shape[0])
    index[:train.shape[0]] = -1
    ps = PredefinedSplit(test_fold=index)

    grid_search = GridSearchCV(
        model, 
        parameters, 
        cv=ps, 
        n_jobs=-1, 
        verbose=1,
        scoring='roc_auc')
    grid_search = grid_search.fit(data[predictors], 
                                  data['label'])
    
    return grid_search



if __name__ == "__main__":
    dfoff = pd.read_csv("ccf_offline_stage1_train.csv")
    dftest = pd.read_csv('ccf_offline_stage1_test_revised.csv')
    dfoff["label"]= dfoff.apply(get_label, axis=1)
    dfoff = coupon_feature(dfoff)
    dftest =coupon_feature(dftest)
    #正确的做法应该是划分数据集，而且至少要划分二分，第一份和第二份时间相邻，作为交叉预测，每一份都包含训练数据集和特征集
    #[07.01,07.30]
    dataset3 = dftest
    #[03.15, 06.30]
    feature3 = dfoff[((dfoff.Date >= "20160315") & (dfoff.Date <="20160630")) | ((dfoff.Date == "null") &((dfoff.Date_received >= "20160315") & (dfoff.Date_received<="20160630")))]
    #

    #[05.15, 06.15]
    dataset2 = dfoff[(dfoff.Date_received >= "20160515") & (dfoff.Date_received <="20160615")]
    feature2 = dfoff[((dfoff.Date >= "20160201") & (dfoff.Date <="20160514")) | ((dfoff.Date == "null") &((dfoff.Date_received >= "20160201") & (dfoff.Date_received<="20160514")))]

    #[04.14, 05.14]
    dataset1 = dfoff[(dfoff.Date_received >= "20160414") & (dfoff.Date_received <="20160514")]
    feature1 = dfoff[((dfoff.Date >= "20160101") & (dfoff.Date <="20160413")) | ((dfoff.Date == "null") &((dfoff.Date_received >= "20160101") & (dfoff.Date_received<="20160413")))]
    print("d1:", dataset1.shape)
    print("f1:", feature1.shape)
    print("d2:", dataset2.shape)
    print("f2:", feature2.shape)
    print("d3:", dataset3.shape)
    print("f3:", feature3.shape)

    dataset1 = featureProcess(feature1, dataset1, dataset1)
    dataset2 = featureProcess(feature2, dataset2, dataset2)
    dataset3 = featureProcess(feature3, dataset3, dataset3)

    #
    dataset1.to_csv("data/dataset1.csv")
    dataset2.to_csv("data/dataset2.csv")
    dataset3.to_csv("data/dataset3.csv")

    predictors = [x for x in dataset1.columns.tolist()if x.startswith("c_") or x.startswith("u_") or x.startswith("m_") or x.startswith("um_") or x.startswith("t_")]

    #使用dataset1和dataset2做交叉验证,验证集使用固定的dataset2
    model = check_model(dataset1, dataset2, predictors)
    print(model.best_score_)
    print(model.best_params_)
    #save
    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    dataset2['pred_prob'] = model.predict_proba(dataset2[predictors])[:,1]
    validgroup = dataset2.groupby(['Coupon_id'])
    aucs = []
    for i in validgroup:
        tmpdf = i[1] 
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    print(np.average(aucs))

