import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() 
%matplotlib inline
import codecs
import os
import codecs
import datetime as dt
import sklearn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import r2_score
import random
from pandas.plotting import scatter_matrix
import itertools
from sklearn import linear_model
clf = linear_model.LinearRegression()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm  
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.metrics import r2_score
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from sklearn import linear_model
clf = linear_model.LinearRegression()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
#-----------------------------------------------
#クラスをインポート　Importing classes
import make_self_data_and_pass_1 as make_self_data

#ここでselfの値を定義する
u = my_di("./20201218_output/data_day_1.csv","./20201218_output/")
#データを書き込むpass_o
pass_o_name=u.pass_o()
#データを読み込む
df_r=u.imp_da()
#dt_aというテストデータのもとと　特徴量のもとを日付で結合する両方の日付を残す
df_a=u.make_da()
#全ての特徴量を利用する際に以下のdfで利用すればよい
df=df_a
print(df)
#-----------------------------------------------
def pick_up_data_2(df_a,df_all_reg,num1,num2,num3):
        endoscopy_3=df_a
        name_all=df_all_reg['Name'].values
        top10_endo=df_all_reg
        print("top10_endo",top10_endo)
        top10_endo.to_csv(r""+u.pass_o()+'top10_endo.csv', encoding = 'shift-jis')
        name_all=top10_endo['Name'].values
        top1=name_all[-num1]
        print("top1",top1)
        top1_v= endoscopy_3[top1].values
        top2=name_all[-num2]
        top2_v= endoscopy_3[top2].values
        top3=name_all[-num3]
        top3_v= endoscopy_3[top3].values
        target= endoscopy_3['endoscopy'].values
        #print(top1_v)
        top10_endoscope=pd.DataFrame({'endoscopy':target,
        str(top1):top1_v,
        str(top2):top2_v,
        str(top3):top3_v})
        df_all=top10_endoscope
        return df_all
#特徴量の回帰係数順にpandas にしたものを取り出す関数この3つが重要　次にlightGBMへ入れる
df_all_reg=u.make_top_all_pandas_not_onehot()
all=pick_up_data_2(df_a,df_all_reg,1,2,3)
all
#-----------------------------------------------

import make_self_light_GBM_bi as light_GBM_bi

#AIdataは今回機械学習のlight GBMに入れたのですべての特徴量で評価
bt = execution_AI_lab(all,u.pass_o()+"lighatGBM_bee_act_output/",0.8,0.3)
#新しくフォルダ作る
bt.pass_out_new()
#データを読み込む
X_train,X_test,X_valid,y_valid, y_train, y_test = bt.advance_preparation()
#そのままlightgbm という機械学習モデルで2値分類実行する関数
bt.lightgbm_binary_classification()
#optunaでパラメータ調整してからlightgbm という機械学習モデルで2値分類実行する関数
auc=bt.lightgbm_binary_classification_optuna()


#-----------------------------------------------
#2021 01 22人工蜂コロニーに入れるための関数作成
def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]

ll=[]
for i in range(100):
    ll.append(i)
ll

#-----------------------------------------------
#bee のfunc関数に入れるための関数 3次元bee_特徴量3の時用
def bee_func_feature_value_auc_3(xin):
    #xinの列を取り出す0列目のみ
    x1=xin[0]
    list1 = ll
    #listに近い値を入れる
    test_bee1=getNearestValue(list1, x1)
    #xinの列を取り出す1列目のみ
    x2=xin[1]
    list2 = ll
    #listに近い値を入れる
    test_bee2=getNearestValue(list2, x2)
    #xinの列を取り出す1列目のみ
    x3=xin[2]
    list3 = ll
    #listに近い値を入れる
    test_bee3=getNearestValue(list3, x3)
    all=pick_up_data_2(df_a,df_all_reg,test_bee1,test_bee2,test_bee3)
    #AIdataは今回機械学習のlight GBMに特徴量で評価
    bt = execution_AI_lab(all,u.pass_o()+"lighatGBM_bee_act_output/",0.6,0.3)
    #データを読み込む
    X_train,X_test,X_valid,y_valid, y_train, y_test = bt.advance_preparation()
    #optunaでパラメータ調整してからlightgbm という機械学習モデルで2値分類実行する関数
    test_auc=bt.lightgbm_binary_classification_optuna()
    out_put=1-test_auc
    return out_put

#-----------------------------------------------
# ABCの実行
import matplotlib.pyplot as plt
import math
from matplotlib import animation

# 関数の設定
# xはnp.array

#sphere
def func(x):
    return np.sum(x**2)

# 初期設定
N = 10# 個体数
d = 3 # 次元
TC = np.zeros(N) #更新カウント
lim = 30
xmax = 100
xmin = 0
G = 3 # 繰り返す回数

x = np.zeros((N,d))
for i in range(N):
    x[i] = (xmax-xmin)*np.random.rand(d) + xmin

# ルーレット選択用関数
def roulette_choice(w):
    tot = []
    acc = 0
    for e in w:
        acc += e
        tot.append(acc)

    r = np.random.random() * acc
    for i, e in enumerate(tot):
        if r <= e:
            return i

x_best = x[0] #x_bestの初期化
best = 100

# 繰り返し
fig = plt.figure()
best_value = []
ims = []
for g in range(G):

    centroid = 1

    def func(x):
        print("x",x)
        out_put1=bee_func_feature_value_auc_3(x)
        out_put1_o=1-out_put1
        return out_put1_o

    best_value.append(best)

    z = np.zeros(N)
    for i in range(N):
        z[i] = func(x[i])


    img2 = plt.scatter(centroid,centroid,color='black')
    img1 = plt.scatter(x[:,0],x[:,1],marker='h',color='gold')
    ims.append([img1]+[img2])


    # employee bee step
    for i in range(N):
        v = x.copy()
        k = i
        while k == i:
            k = np.random.randint(N)


        for j in range(d):
            r = np.random.rand()*2-1 #-1から1までの一様乱数
            v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])

        if func(x[i]) > func(v[i]):
            x[i] = v[i]
            TC[i] = 0
        else: TC[i] += 1

    # onlooker bee step
    for i in range(N):
        w = []
        for j in range(N):
            w.append(np.exp(-func(x[j])))
        i = roulette_choice(w)
        for j in range(d):
            r = np.random.rand()*2-1 #-1から1までの一様乱数
            v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])
        if func(x[i]) > func(v[i]):
            x[i] = v[i]
            TC[i] = 0
        else: TC[i] += 1



    # scout bee step
    for i in range(N):
        if TC[i] > lim:
            for j in range(d):
                x[i,j] = np.random.rand()*(xmax-xmin) + xmin
            TC[i] = 0

    # 最良個体の発見
    for i in range(N):
        if best > func(x[i]):
            x_best = x[i]
            best = func(x_best)



# 結果
print(x_best,func(x_best))

ani = animation.ArtistAnimation(fig, ims, interval=70)
plt.show()
