
#dayを曜日データで数字にする　ソースコード
def Sort_by_date(send2_allday):
    s_date=send2_allday
    df_date=s_date.reset_index()
    df_date['day'] = pd.to_datetime(df_date['day'])
    df_date['曜日']=df_date['day'].apply(lambda x:x.weekday())
    #日付順に並び替える
    sorted_df = df_date.sort_values(['day'])
    sorted_df2p=sorted_df.reset_index()
    sorted_df2=sorted_df2p.drop('index', axis=1)
    sorted_df2
    #datetime　からstr変換
    df_date=sorted_df2
    df_date['day'] = pd.to_datetime(df_date['day'])
    df_date['day'] = df_date['day'].astype(str)

    sorted_df3=df_date
    #dayの('-', '/')を入れ替えたいので以下の方法を利用した
    df1 = sorted_df3

    sorted_df3r=df1['day'].str.replace('-', '/')
    sorted_df3r
    df1_non_day=df1.drop('day', axis=1)
    df1_non_day['day']=sorted_df3r
    sorted_df3_d=df1_non_day
    sorted_df3_d
    return sorted_df3_d

sorted_df3_2nd=Sort_by_date(send2_allday)
sorted_df3_2nd
