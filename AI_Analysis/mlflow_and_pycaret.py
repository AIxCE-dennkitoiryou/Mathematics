#mlflowとは、機械学習のライフサイクル(前処理→学習→デプロイ)を管理するオープンソースなプラットフォーム
#Tracking: ロギング
#ロギング（英：logging）とは
#「取るぜ取るぜ～ログ取るぜ～」のこと。
#実際にロギングを行ってみる ☛　pycaret の後に実施
#mlflow is an open source platform to manage the machine learning lifecycle (preprocessing -> training -> deployment)
#Tracking: Logging
#Logging (English: logging) means.
#"Take it, take it, take it - log it.
#Actual logging ☛ Performed after pycaret

mlflow.set_tracking_uri('./hoge/mlruns/')

# experimentが存在しなければ作成される。
mlflow.set_experiment('compare_max_depth')

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
import datetime
from sklearn.model_selection import train_test_split

class my_directory_p:
    def __init__(self,pass_out):
        #self.day_im = day_im
        self.pass_out = pass_out


    def pass_o(self):
        return self.pass_out

    def pass_out_new(self):
        # フォルダ「output」が存在しない場合は作成する
        data_dir = self.pass_out
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print(u.pass_o()+"フォルダ作成しました！！")
        
    def imp_data(self):
        #ubuntu 
        image_file_path = './data/"data_name".csv'

        import pandas as pd

        with codecs.open(image_file_path, "r", "Shift-JIS", "ignore") as file:
                df_r9 = pd.read_table(file, delimiter=",")

        return df_r9
        
#ここでselfの値を定義する
u = my_directory_p("./20210201_output/")
u.pass_out_new()
df_r=u.imp_data()

u.pass_o()

#----------------------

#data predect データをpandasで準備する

#----------------------
#mlflowとは、機械学習のライフサイクル(前処理→学習→デプロイ)を管理するオープンソースなプラットフォーム
#Tracking: ロギング
#ロギング（英：logging）とは

#「取るぜ取るぜ～ログ取るぜ～」のこと。
#実際にロギングを行ってみる ☛　pycaret の後に実施
with mlflow.start_run():
    mlflow.log_param('param1', 1) # パラメータ
    mlflow.log_metric('metric1', 0.1) # スコア
    mlflow.log_artifact(filename) # その他、モデルやデータなど
mlflow.search_runs() # experiment内のロギング内容を取得できる

#------------------------------

#anaconda prompt を開く　Open the anaconda prompt

#URIで設定したディレクトリまで移動する。 
#この時、 mlruns ディレクトリが配下になるようにする( mlruns ディレクトリが存在しない場合、 mlruns ディレクトリが作成される)。 
#mlflow ui でローカルサーバが起動する。

#Go to the directory set in URI. 
#Move to the directory set in the URI. At this time, make sure that the mlruns directory is under it (if the mlruns directory does not exist, the mlruns directory is created). 
#The local server is started by mlflow ui.


#𝑐𝑑./ℎ𝑜𝑔𝑒/  ls mlruns
$ mlflow ui
#------------------------------

#ブラウザ上で http://127.0.0.1:5000 を開く
#------------------------------

tracking = mlflow.tracking.MlflowClient()
experiment = tracking.get_experiment_by_name('hoge')

#pycaret and mlflow  https://pycaret.org/mlflow/

# tracking uri 
import mlflow 
mlflow.set_tracking_uri('./hoge/mlruns/')

from pycaret.regression import *
exp_name = setup(df_all, target = 'target',train_size = 0.99,silent=True,fold_strategy='timeseries',data_split_shuffle=False, log_experiment = True, experiment_name = 'diabetes1')
best = compare_models()

omp= create_model("lightgbm")
#plot_model(omp)
evaluate_model(omp)

omp= create_model("lightgbm",fold = 5)
evaluate_model(omp)
pred_unseen = predict_model(omp)
pred_unseen
pred_unseen.to_csv(r""+u.pass_o()+'lightgbm_test7_top10_data.csv', encoding = 'shift-jis')
omp= create_model("lightgbm",cross_validation=False)
#evaluate_model(omp)
tuned_lr = tune_model(omp)
evaluate_model(tuned_lr)
pred_unseen = predict_model(tuned_lr)
pred_unseen
pred_unseen.to_csv(r""+u.pass_o()+'lightgbm_tuned_test7_top10_data.csv', encoding = 'shift-jis')
