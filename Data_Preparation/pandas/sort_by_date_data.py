#日付データでソートをかけたいときはこれを利用Use this to sort by date data.

send2=df_send_data_total

send2_d=send2.reset_index()
#print(send2_d)
day1_m=send2_d["day1"]
#print(day1_m)
#date time に変更した！！！！

str_date =send2_d["day1"].astype(str)
#date time に変更したい
#文字列を削除してくっつけたい
dpc_y = send2_d["day1"].str[:4]
dpc_m = send2_d["day1"].str[5:7]
dpc_d = send2_d["day1"].str[8:10]
#print(type(dpc_m))
#print(dpc_d)

dpc_date_p= dpc_y + '-' + dpc_m + '-'+ dpc_d

#print(dpc_date)
dpc_date=pd.to_datetime(dpc_date_p)
print(dpc_date)
