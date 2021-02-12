#pandasの中のcolumns（target）をn_day分ずらす変数　これによってn_day先を予測することができるfor文で回すことで調節が可能
#variable that shifts columns (target) in pandas by n_days This allows us to predict n_days ahead, and can be adjusted by turning it with a for statement.
def future_prediction_day(dfdata,culum_u,n_day):

        def Create_Description_X(dtt):
            train_data = dtt
            X = train_data.drop(culum_u, axis=1)# 削除
            #説明変数作成
            #print("X.shape",X.shape)
            return X
        #説明変数Create_Description_X実施これでXに値が入るはず
        Xp = Create_Description_X(dfdata)
        #print("Create_Description_X 後の X.shape",X.shape)
        #目的変数入力用関数
        def Objective_variable_creationY(Ymoto):
            targek = Ymoto
            Y=targek.T
            return Y
        #目的変数作成
        targetk = dfdata[culum_u].values
        #目的変数作成関数利用
        Yp = Objective_variable_creationY(targetk)
        #pandsにもどす
        Y = pd.DataFrame({'target':Yp})
        #pandasに戻す
        X = pd.DataFrame(Xp)
        #目的変数の最初の行をn_day分削除
        #print(Y)
        Y_future = Y[n_day:]
        print(n_day,"日分削除した目的変数↓")
        print("Y_future",Y_future)
        Y_future2 =Y_future.reset_index()
        #説明変数の最後の行をn_day分削除
        X_future = X[:-n_day]
        X_future2 =X_future.reset_index()       
        Y_X_future2 =pd.concat([Y_future2, X_future2], axis=1)
        Y_X_future2
        Y_X_future2_o=Y_X_future2
        return Y_X_future2_o

test1 = future_prediction_day(df_one_hot_encoded,"target",1)
print(test1)
test2 = future_prediction_day(send2,"target",2)
print(test2)
test3 = future_prediction_day(send2,"target",3)
print(test3)
test4 = future_prediction_day(send2,"target",4)
print(test4)
test5 = future_prediction_day(send2,"target",5)
print(test5)
test6 = future_prediction_day(send2,"target",6)
print(test6)
test7 = future_prediction_day(send2,"target",7)
print(test7)

#----------------------

#予想したい日付後の値を選択　Select the value after the date you want to predict.
# tes の変数の中で"target"が一番左にあることが前提 Assume that "target" is the leftmost variable in tes.
tes=test1.drop("index", axis=1)# 削除

#----------------------
#tes の変数の中で"target"が一番左にあることが前提で回帰のベースライン表示　Display the baseline of the regression assuming that "target" is the leftmost variable in tes.
def make_Base_line(df,numb):
	import pandas as pd
	import numpy as np
	# Mean Absolute Error(MAE)用
	from sklearn.metrics import mean_absolute_error
	# Root Mean Squared Error(RMSE)
	from sklearn.metrics import mean_squared_error
	#以下のでカラムの値を取り出して名前を入れれる You can extract the value of the column and put the name in it as follows
	columns_name = df.columns
	#print(columns_name)
	Serial_number= [x for x in range(len(columns_name))]
	#print("Serial_number",Serial_number)
	#print(columns_name[numb])
	mean_1=df[columns_name[numb]].mean()
	#print("columns_name and mean",mean_1)
	lst = [df[columns_name[numb]].mean()] * len(df.index)     
	## label data
	label = df[str(columns_name[numb])]
	## AI predicted data
	pred = lst
	# MAE計算
	mae = mean_absolute_error(label, pred)
	#print('MAE : {:.3f}'.format(mae))
	# {:.3f}で小数点以下は3桁で表示
	# RMSE計算
	rmse = np.sqrt(mean_squared_error(label, pred))
	#print('RMSE : {:.3f}'.format(rmse))
	index1 = ["mean", 'MAE_mean', 'RMSE_mean']
	columns1 =[str(columns_name[numb])]
	Calculation=pd.DataFrame(data=[df[columns_name[numb]].mean(),format(mae),format(rmse)], index=index1, columns=columns1)
	#print(Calculation)
	return Calculation

Calculation=make_Base_line(tes,0)
Calculation
#pycaretの実施
from pycaret.regression import *
exp_name = setup(tes, target = 'target',train_size = 0.99,silent=True,fold_strategy='timeseries',data_split_shuffle=False)
best = compare_models()
#----------------------
omp= create_model("lightgbm",fold = 5)
evaluate_model(omp)
pred_unseen = predict_model(omp)
pred_unseen
#----------------------
omp= create_model("lightgbm",cross_validation=False)
#evaluate_model(omp)
tuned_lr = tune_model(omp)
evaluate_model(tuned_lr)
pred_unseen = predict_model(tuned_lr)
pred_unseen

#----------------------

def make_top10_not_onehot(dt):
        print(dt)
        #説明変数入力用ここをうまく少ないソースコードで取得できる方法を考案
        def Create_Description_X(dtt):
         train_data = dtt
         X = train_data.drop("target", axis=1)# 削除
        
         #説明変数作成
         print("X.shape",X.shape)
         return X

        #説明変数Create_Description_X実施これでXに値が入るはず
        X = Create_Description_X(dt)
        #print("Create_Description_X 後の X.shape",X.shape)

        #目的変数入力用関数
        def Objective_variable_creationY(Ymoto):
         targek = Ymoto
         Y=targek.T
         return Y

        #目的変数作成
        targetk = dt["target"].values
        #目的変数作成関数利用
        Y = Objective_variable_creationY(targetk)

        #print("Objective_variable_creationY 後の Y.shape",Y.shape)

        # 訓練用のデータと、テスト用のデータに分ける関数
        def Test_data_and_training_data_split(df,X,Y):
         N_train = int(len(df) * 0.90)
         N_test = len(df) - N_train
         X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=N_test,shuffle=False)
         return X_train, X_test, y_train, y_test
        # 訓練用のデータと、テスト用のデータに分ける関数実行
        X_train, X_test, y_train, y_test = Test_data_and_training_data_split(dt,X,Y)
        endoscopy_3=dt
        send2=dt
        send2_p=send2
        #send2_p = send2_p[:-1]
        wine_except_quality = send2_p.drop("target", axis=1)
        # sklearn.linear_model.LinearRegression クラスを読み込み
        from sklearn import linear_model
        clf = linear_model.LinearRegression()
            # 説明変数に X_trainを利用
        X = X_train
            # 目的変数に Y_train を利用
        Y = y_train
            # 予測モデルを作成
        clf.fit(X, Y)
            # 偏回帰係数
        print(pd.DataFrame({"Name":wine_except_quality.columns,
        "Coefficients":clf.coef_}).sort_values(by='Coefficients') )
            # 切片 (誤差)
        #print(clf.intercept_)
        top10_endo=pd.DataFrame({"Name":wine_except_quality.columns,
        "Coefficients":clf.coef_}).sort_values(by='Coefficients')
        name_all=top10_endo['Name'].values

        print(top10_endo)

        name_all=top10_endo['Name'].values
        print("name_all",name_all)
        top1=name_all[-1]
        top1_v= endoscopy_3[top1].values
        top2=name_all[-2]
        top2_v= endoscopy_3[top2].values
        top3=name_all[-3]
        top3_v= endoscopy_3[top3].values
        top4=name_all[-4]
        top4_v= endoscopy_3[top4].values
        top5=name_all[-5]
        top5_v= endoscopy_3[top5].values
        top6=name_all[-6]
        top6_v= endoscopy_3[top6].values
        top7=name_all[-7]
        top7_v= endoscopy_3[top7].values
        top8=name_all[-8]
        top8_v= endoscopy_3[top8].values
        top9=name_all[-9]
        top9_v= endoscopy_3[top9].values
        top10=name_all[-10]
        top10_v= endoscopy_3[top10].values
        target= endoscopy_3["target"].values
        print(target)
        print(top1_v)
        print(top2_v)
        top10_endoscope=pd.DataFrame({"target":target,
        str(top1):top1_v,
        str(top2):top2_v,
        str(top3):top3_v,
        str(top4):top4_v,
        str(top5):top5_v,
        str(top6):top6_v,
        str(top7):top7_v,
        str(top8):top8_v,
        str(top9):top9_v,
        str(top10):top10_v})
        #print(top10_endoscope)
        df_all=top10_endoscope

        return df_all  
df_all=make_top10_not_onehot(tes)

from pycaret.regression import *
exp_name = setup(df_all, target = 'target',train_size = 0.99,silent=True,fold_strategy='timeseries',data_split_shuffle=False)
best = compare_models()
 
omp= create_model("lightgbm",fold = 5)
evaluate_model(omp)
pred_unseen = predict_model(omp)
pred_unseen

omp= create_model("lightgbm",cross_validation=False)
#evaluate_model(omp)
tuned_lr = tune_model(omp)
evaluate_model(tuned_lr)
pred_unseen = predict_model(tuned_lr)
pred_unseen
