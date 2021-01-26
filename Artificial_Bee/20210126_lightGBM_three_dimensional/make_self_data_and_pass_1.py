class my_di:
    def __init__(self,pass_i,pass_out):
        #読み込みたいフォルダデータを入れるところです！
        self.pass_i = pass_i
        #書き出したいフォルダデータを入れるところです！
        self.pass_out = pass_out
    def pass_o(self):
        return self.pass_out
    def imp_da(self):
        #ubuntu 
        image_file_path = self.pass_i
        import pandas as pd
        with codecs.open(image_file_path, "r", "Shift-JIS", "ignore") as file:
                df = pd.read_table(file, delimiter=",")
        print(df)
        return df
    def make_da(self):
        #ubuntuの場合以下を調整する必要がある
        with codecs.open(r"#pass#", "r", "Shift-JIS", "ignore") as file:
          d_p_spd_endoscopy = pd.read_table(file, delimiter=",")
        #day1  と　endoscopy　の２つを取り出す
        d_p_endoscopy=d_p_spd_endoscopy.loc[:, ['day1', 'endoscopy']]
        endoscopy_o=d_p_spd_endoscopy.loc[:, ['endoscopy']]
        send2_d=d_p_endoscopy
        day1_m=send2_d["day1"]
        #date time に変更した！！！！
        str_date =send2_d["day1"].astype(str)
        #date time に変更したい
        #文字列を削除してくっつけたい
        dpc_y = send2_d["day1"].str[:4]
        dpc_m = send2_d["day1"].str[5:7]
        dpc_d = send2_d["day1"].str[8:10]
        dpc_date_p= dpc_y + '-' + dpc_m + '-'+ dpc_d
        e_date=pd.to_datetime(dpc_date_p)
        #ここで日付とデータを横に着ける
        d_e =pd.concat([e_date, endoscopy_o], axis=1)
        #ここで準備した内視鏡データとDPCデータを日付でくっつける
        d_e_p=d_e
        #pandas のdatetime を文字列に　astype　する関数
        def datetime_astype(d_e):
            d_e2 =d_e["day1"].astype(str)
            d_e3=d_e.drop("day1", axis=1)
            d_e_n = pd.concat([d_e2,d_e3], axis=1)
            return d_e_n
        #関数の実行でまず day1の値を　文字列に
        d_e_n=datetime_astype(d_e_p)
        #関数の実行でDPCの day1の値を　文字列に
        df_r2=datetime_astype(df_r)
        #文字列に変換したのでmargeを行えるhow=inner にすることで　一致しない日付は消す
        dt_a = pd.merge(d_e_n, df_r2, on='day1', how='inner')
        #日付を残した値を入れたいときはdt_aを使う
        #print(dt_a)
        endoscopy_2=dt_a.drop("day1", axis=1)
        endoscopy_2.dtypes
        #以下のソースコードで全て０の列は消す
        endoscopy_3 = endoscopy_2.loc[:, (endoscopy_2 != 0).any(axis=0)]
        #print(endoscopy_3)
        df_all_t=endoscopy_3  
        #return dt_a  #日付を残した値を入れたいときはdt_aを使う
        return df_all_t
    
    def make_top10(self):
        print(df_a)
        #説明変数入力用ここをうまく少ないソースコードで取得できる方法を考案
        def Create_Description_X(dt):
         train_data = dt.values
         X = train_data[:, 1:] # 2列目以降の変数
         y  = train_data[:, 0]  # 正解データを1列目に置きましたそしてyとしました
         #説明変数作成
         print("X.shape",X.shape)
         return X

        #説明変数Create_Description_X実施これでXに値が入るはず
        X = Create_Description_X(df_a)
        #print("Create_Description_X 後の X.shape",X.shape)

        #目的変数入力用関数
        def Objective_variable_creationY(Ymoto):
         targek = Ymoto
         Y=targek.T
         return Y

        #目的変数作成
        targetk = df['endoscopy'].values
        #目的変数作成関数利用
        Y = Objective_variable_creationY(targetk)

        #print("Objective_variable_creationY 後の Y.shape",Y.shape)

        # 訓練用のデータと、テスト用のデータに分ける関数
        def Test_data_and_training_data_split(df,X,Y):
         N_train = int(len(df) * 0.60)
         N_test = len(df) - N_train
         X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=N_test,shuffle=False)
         return X_train, X_test, y_train, y_test
        # 訓練用のデータと、テスト用のデータに分ける関数実行
        X_train, X_test, y_train, y_test = Test_data_and_training_data_split(df,X,Y)

        r_data=[]
        date=[]
        code_d_s=[]
        count_s=[]
        y1=[]
        y2=[]
        y3=[]

        endoscopy_3=df_a
        send2=df_a
        send2_p=send2
        send2_p = send2_p[:-1]
        wine_except_quality = send2_p.drop('endoscopy', axis=1)
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
        top10_endo.to_csv(r""+u.pass_o()+'top10_endo.csv', encoding = 'shift-jis')
        name_all=top10_endo['Name'].values
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
        target= endoscopy_3['endoscopy'].values
        #print(top1_v)
        top10_endoscope=pd.DataFrame({'endoscopy':target,
        "top1_v":top1_v,
        "top2_v":top2_v,
        "top3_v":top3_v,
        "top4_v":top4_v,
        "top5_v":top5_v,
        "top6_v":top6_v,
        "top7_v":top7_v,
        "top8_v":top8_v,
        "top9_v":top9_v,
        "top10_v":top10_v})
        #print(top10_endoscope)
        df_all=top10_endoscope
        # リストの作成
        one_hot_list = [ "top1_v",  "top2_v",  "top3_v",  "top4_v",  "top5_v",  "top6_v" , "top7_v",  "top8_v", "top9_v",  "top10_v"]

        # one-hot encodingの実施
        df_all_1 = pd.get_dummies(df_all, columns = one_hot_list)
        dftop10=df_all_1
        
        return dftop10
    def make_da(self):
        #ubuntuの場合以下を調整する必要がある
        with codecs.open(r"#pass#", "r", "Shift-JIS", "ignore") as file:
          d_p_spd_endoscopy = pd.read_table(file, delimiter=",")
        #day1  と　endoscopy　の２つを取り出す
        d_p_endoscopy=d_p_spd_endoscopy.loc[:, ['day1', 'endoscopy']]
        endoscopy_o=d_p_spd_endoscopy.loc[:, ['endoscopy']]
        send2_d=d_p_endoscopy
        day1_m=send2_d["day1"]
        #date time に変更した！！！！
        str_date =send2_d["day1"].astype(str)
        #date time に変更したい
        #文字列を削除してくっつけたい
        dpc_y = send2_d["day1"].str[:4]
        dpc_m = send2_d["day1"].str[5:7]
        dpc_d = send2_d["day1"].str[8:10]
        dpc_date_p= dpc_y + '-' + dpc_m + '-'+ dpc_d
        e_date=pd.to_datetime(dpc_date_p)
        #ここで日付とデータを横に着ける
        d_e =pd.concat([e_date, endoscopy_o], axis=1)
        #ここで準備したデータを日付でくっつける
        d_e_p=d_e
        #pandas のdatetime を文字列に　astype　する関数
        def datetime_astype(d_e):
            d_e2 =d_e["day1"].astype(str)
            d_e3=d_e.drop("day1", axis=1)
            d_e_n = pd.concat([d_e2,d_e3], axis=1)
            return d_e_n
        #関数の実行でまず内視鏡の day1の値を　文字列に
        d_e_n=datetime_astype(d_e_p)
        #関数の実行でDPCの day1の値を　文字列に
        df_r2=datetime_astype(df_r)
        #文字列に変換したのでmargeを行えるhow=inner にすることで　一致しない日付は消す
        dt_a = pd.merge(d_e_n, df_r2, on='day1', how='inner')
        #日付を残した値を入れたいときはdt_aを使う
        #print(dt_a)
        endoscopy_2=dt_a.drop("day1", axis=1)
        endoscopy_2.dtypes
        #以下のソースコードで全て０の列は消す
        endoscopy_3 = endoscopy_2.loc[:, (endoscopy_2 != 0).any(axis=0)]
        #print(endoscopy_3)
        df_all_t=endoscopy_3  
        #return dt_a  #日付を残した値を入れたいときはdt_aを使う
        return df_all_t
    def make_top_all_pandas_not_onehot(self):
        print(df_a)
        #説明変数入力用ここをうまく少ないソースコードで取得できる方法を考案
        def Create_Description_X(dt):
         train_data = dt.values
         X = train_data[:, 1:] # 2列目以降の変数
         y  = train_data[:, 0]  # 正解データを1列目に置きましたそしてyとしました
         #説明変数作成
         print("X.shape",X.shape)
         return X

        #説明変数Create_Description_X実施これでXに値が入るはず
        X = Create_Description_X(df_a)
        #print("Create_Description_X 後の X.shape",X.shape)

        #目的変数入力用関数
        def Objective_variable_creationY(Ymoto):
         targek = Ymoto
         Y=targek.T
         return Y

        #目的変数作成
        targetk = df['endoscopy'].values
        #目的変数作成関数利用
        Y = Objective_variable_creationY(targetk)

        #print("Objective_variable_creationY 後の Y.shape",Y.shape)

        # 訓練用のデータと、テスト用のデータに分ける関数
        def Test_data_and_training_data_split(df,X,Y):
         N_train = int(len(df) * 0.60)
         N_test = len(df) - N_train
         X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=N_test,shuffle=False)
         return X_train, X_test, y_train, y_test
        # 訓練用のデータと、テスト用のデータに分ける関数実行
        X_train, X_test, y_train, y_test = Test_data_and_training_data_split(df,X,Y)

        r_data=[]
        date=[]
        code_d_s=[]
        count_s=[]
        y1=[]
        y2=[]
        y3=[]

        endoscopy_3=df_a
        send2=df_a
        send2_p=send2
        send2_p = send2_p[:-1]
        wine_except_quality = send2_p.drop('endoscopy', axis=1)
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
 
        return top10_endo          
