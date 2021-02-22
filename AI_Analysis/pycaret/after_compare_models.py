# store score grid in dataframe 
df = pull()
def makelist(df):
    df = pull()
    print(df)
    #indexを取り出すExtract index
    index=df.index
    print(index)
    list_c1=[]
    for i in index:
     print(i)
     list_c1.append(i)
    return list_c1
list_c1=makelist(df)

#-------------------------------
#best modelで予測
omp= create_model(list_c1[0])
#plot_model(omp)
evaluate_model(omp)

pred_unseen = predict_model(omp)

#-------------------------------
#lightgbmで予測
omp= create_model("lightgbm")
#plot_model(omp)
evaluate_model(omp)

pred_unseen = predict_model(omp)
