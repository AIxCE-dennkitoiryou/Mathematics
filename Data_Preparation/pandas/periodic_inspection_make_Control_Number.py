import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import codecs
import os


class my_o_directory:
    def __init__(self,pass_out):
        self.pass_out = pass_out
    def print_name(self):
        print(self.pass_out)
    def pass_o(self):
        return self.pass_out
    def pass_out_new(self):
        # フォルダ「output」が存在しない場合は作成する
        data_dir = self.pass_out
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print(u.pass_o()+"フォルダ作成しました！！")

    def imp_data(self):
        #送信データを追加したいときはこちらを追加したのちに　dfpp_m_send〇=open_send_data(image_file_path〇)
        #と〇を更新してからさらにマージさせてください
        #------------------↓1つ目のデータ------------------
        image_file_path = './data/2020_periodic_inspection_tk_HP.csv'
        #image_file_path2 = './data/df_all_spd_day.csv'

        #masuta dataを変えたい場合は以下の名前を変えてください
        def open_send_data(image_file_path_v):
            with codecs.open(image_file_path_v, "r", "Shift-JIS", "ignore") as file:
                    dfpp = pd.read_table(file, delimiter=",")
            dfpp_m_send = dfpp
            return dfpp_m_send
        hituyoudo1=open_send_data(image_file_path)
        print("hituyoudo1",hituyoudo1)
        df1_2_o = hituyoudo1
        #------------------↓2つ目のデータ------------------
        image_file_path2 = './data/2021_periodic_inspection_tk_HP_test.csv'
        #image_file_path2 = './data/df_all_spd_day.csv'
        #masuta dataを変えたい場合は以下の名前を変えてください
        def open_send_data(image_file_path_v):
            with codecs.open(image_file_path_v, "r", "Shift-JIS", "ignore") as file:
                    dfpp = pd.read_table(file, delimiter=",")
            dfpp_m_send = dfpp
            return dfpp_m_send
        teiki2=open_send_data(image_file_path2)
        print("hituyoudo1",hituyoudo1)
        df1_2_o2 = teiki2
        return df1_2_o,df1_2_o2

#ここでselfの値を定義する 定期点検
u = my_o_directory("./202103_periodic_inspection/")
u.print_name()
u.pass_out_new()
df1_2_o,df1_2_o2=u.imp_data()


all_data=pd.merge(df1_2_o,df1_2_o2, on='管理番号', how='outer')

all_data.to_csv(r''+u.pass_o()+"periodic_inspection"+'.csv', encoding = 'shift-jis')
