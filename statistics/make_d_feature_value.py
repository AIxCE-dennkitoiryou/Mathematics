def make_d_feature_value(df,numb):
    #以下のでカラムの値を取り出して名前を入れれる You can extract the value of the column and put the name in it as follows
    columns_name = df.columns
    print(columns_name)
    Serial_number= [x for x in range(len(columns_name))]
    #print("Serial_number",Serial_number)
    #print("target_del_columns_name☛",columns_name[numb])
    df_d=df.drop(str(columns_name[numb]), axis=1)
    return df_d
    
#以下の関数で0番目のカラムを削除する    
df_d = make_d_feature_value(df,0)
print(df_d)
