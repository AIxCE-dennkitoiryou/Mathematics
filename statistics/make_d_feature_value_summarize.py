def make_d_feature_value_summarize(df,n1,n2,n3,n4):
    #まとめて特徴量を減らしたい場合上の,n1,n2,n3,n4 に値を追加していく左の例では4つ削除したいときに使う
    #以下のでカラムの値を取り出して名前を入れれる You can extract the value of the column and put the name in it as follows まとめる
    columns_name = df.columns
    print(columns_name)
    Serial_number= [x for x in range(len(columns_name))]
    #print("Serial_number",Serial_number)
    #print("target_del_columns_name☛",columns_name[numb])
    #まとめて特徴量を減らしたい場合↓columns_name[n1]の値を追加していく
    df_d=df.drop([str(columns_name[n1]),str(columns_name[n2]),str(columns_name[n3]),str(columns_name[n4])], axis=1)
    return df_d
    
df_a = make_d_feature_value_summarize(df,22,21,20,19)
print(df_a)
