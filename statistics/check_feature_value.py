def check_feature_value(df):
    #特徴量のカラムの数字と名前を確認したいときに使う変数
    columns_name = df.columns
    Serial_number= [x for x in range(len(columns_name))]
    print(columns_name)
    print("Serial_number",Serial_number)
    for bp in list(zip(Serial_number,columns_name)):
        print(bp)
check_feature_value(df)
