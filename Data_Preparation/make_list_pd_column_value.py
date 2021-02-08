def make_pd_column_value(df_b,column_v):
    df_b_1 = df_b
    df_b_1_p = df_b_1[column_v].values
    dpc_d_p = df_b_1_p.tolist()
    #以下のソースコードで重なる値をを消す Eliminate overlapping values in the following source code
    dpc_d = list(dict.fromkeys(dpc_d_p))
    #print(dpc_d)
    dpc=[]
    for i in dpc_d:
     #print(i)
     dpc.append(i)
    np.nan_to_num(dpc)
    for i in dpc:
     Department_name = i
     print(i)
    return dpc
