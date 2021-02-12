    def make_top10_regression_coefficient(dt):
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
