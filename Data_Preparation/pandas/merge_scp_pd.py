def merge_scp_pd(scope_df,num1,num2):
    df3 = merge_pd(scope_df,scp[num1]) 
    df4 = merge_pd(scope_df,scp[num2])
    df5=pd.merge(df4,df3,on=["day"], how='outer')
    return df5
df5=merge_scp_pd(scope_df,2,3)
df6=merge_scp_pd(scope_df,4,5)
df5_6=pd.merge(df5,df6,on=["day"], how='outer')
df7=merge_scp_pd(scope_df,6,7)
df8=merge_scp_pd(scope_df,8,9)
df7_8=pd.merge(df7,df8,on=["day"], how='outer')
dfall=pd.merge(df5_6,df7_8,on=["day"], how='outer')
dfall
