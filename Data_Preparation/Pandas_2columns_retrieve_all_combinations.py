#Pandas の"day"と'count'の2つの　カラム　をリスト化して すべての組み合わせを取り出す方法
#How to list the two columns "day" and "count" in Pandas and retrieve all combinations
def make_itertools_department(send_dt):
         #マルチインデックス解除
         send_dt1=send_dt.reset_index()
         send_dy_p = send_dt1["day"].values
         send_dy1 = send_dy_p.tolist()
         send_dy = list(dict.fromkeys(send_dy1))
         
         send_dy_p = send_dt['count'].values
         send_dy3 = send_dy_p.tolist()
         send_dy3 = list(dict.fromkeys(send_dy3))

         import itertools
         import pprint
         t = send_dy
         r = send_dy3
         l_p = list(itertools.product(r,t))
         pprint.pprint(l_p)
         name = ["d_count","day"]
         dt_b_t = pd.DataFrame(l_p,columns=name)
         return dt_b_t
#Used intertool on the whole data
dt_b_t = make_itertools_department(df_one_hot_encoded)
dt_b_t
