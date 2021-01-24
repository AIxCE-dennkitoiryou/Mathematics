


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
    


# import numpy as np
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
