def results_out(target,name):
    prot_df=pred_unseen["Label"].astype(float)
    prot_df
    y_test=pred_unseen[target].values
    y_test1=np.array(y_test, dtype=int).tolist()
    y_test1s = pd.Series(y_test1)
    return prot_df,y_test1sdef Save_the_results(target,name):
    prot_df=pred_unseen["Label"].astype(float)
    prot_df
    y_test=pred_unseen[target].values
    y_pred=prot_df

    from sklearn.metrics import r2_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    y_test1=np.array(y_test, dtype=int).tolist()
    print(y_test1)
    y_pred1=y_pred.tolist()
    print(y_pred1)
    MAE=mean_absolute_error(y_test1, y_pred1)
    print(MAE)
    MSE=mean_squared_error(y_test1, y_pred1)
    print(MSE)
    RMSE=np.sqrt(mean_squared_error(y_test1, y_pred1))
    print(RMSE)
    R2=r2_score(y_test1, y_pred1)
    print(R2)
    # data
    data_dic = {
        target:[name],
        'MAE':[MAE],
        'MSE': [MSE],
        'RMSE': [RMSE],
        'R2': [R2],
    }
    #columns = ['MAE', 'MAE', 'RMSE',"R2"]
    #data=[MAE,MSE,RMSE,R2]

    # DataFrame作成
    df = pd.DataFrame(data=data_dic)
    print(df)
    # データ確認
    df.to_csv(r''+u.pass_o()+"lightgbm_pred_unseen"+name+'.csv', encoding = 'shift-jis')
Save_the_results("target","test1")


def results_out(target,name):
    prot_df=pred_unseen["Label"].astype(float)
    prot_df
    y_test=pred_unseen[target].values
    y_test1=np.array(y_test, dtype=int).tolist()
    y_test1s = pd.Series(y_test1)
    return prot_df,y_test1s
  
from pycaret.regression import *
exp_name = setup(tes, target = 'target',train_size = 0.52,silent=True,fold_strategy='timeseries',data_split_shuffle=False,log_experiment = True, experiment_name = 'Endoscope_Cleaner_test1')
best = compare_models()  

omp= create_model("rf")
#plot_model(omp)
#evaluate_model(omp)
pred_unseen = predict_model(omp)
Save_the_results("target","Endoscope_Cleaner_test1")
prot_df,y_test=results_out("target","Endoscope_Cleaner_test2")
