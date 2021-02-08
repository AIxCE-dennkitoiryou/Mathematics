#scpを利用してグループ化してカウントした値を返す Use scp to group and return counted values

day_l=[]
scope_l=[]
coun_l=[]
no_l=[]

def ward_name(data,colum1,colum2,count_t2):
        df_d = data
        df_1=df_d.reset_index()
        #マルチインデックスを作成　Create a multi-index
        df_2 = df_1.set_index([colum1,colum2])
        #重なる値をを消したものをspcとした場合　要素を iに入力しマルチインデックスで検索 Enter the value of i after removing overlapping values and search with multi index.
        for i in scp:
            #print(i)
            scp_1 = df_2.xs(i, level=0)
            #以下でカウントして
            send_dt_d2 = scp_1.groupby([count_t2]).count()
            #pandas.DataFrameの任意の2列から辞書生成　以下の場合　もしpandasのカラムに'No.'があればそこの値をリストとして取り出す
            #以下のソースコードで
            for n, (dday, coun) in enumerate(zip(send_dt_d2.index,send_dt_d2['No.'])):
                print("n",n)
                print("dday",dday)
                print("coun",coun)
                print("i",i)
                no_l.append(n)
                day_l.append(dday)
                coun_l.append(coun)
                scope_l.append(i)

ward_name(df_b,"カラムの名前1_set_index","カラムの名前2_日付推奨_set_index","カラムの名前3_日付推奨")
print(all_dect)
#dataframeにするためのリストの値を設定　Set the value of the list to be a dataframe.
students = [ no_l ,day_l,coun_l,scope_l]
# Creating a dataframe object from listoftuples
dfObj = pd.DataFrame(students, index= ['index' , 'day', 'count',"scope"]).T
