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

test1 = future_prediction_day(send2,"target",1)
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
