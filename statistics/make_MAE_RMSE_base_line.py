def make_Base_line(df,numb):
	import pandas as pd
	import numpy as np
	# Mean Absolute Error(MAE)用
	from sklearn.metrics import mean_absolute_error
	# Root Mean Squared Error(RMSE)
	from sklearn.metrics import mean_squared_error
	#以下のでカラムの値を取り出して名前を入れれる You can extract the value of the column and put the name in it as follows
	columns_name = df.columns
	#print(columns_name)
	Serial_number= [x for x in range(len(columns_name))]
	#print("Serial_number",Serial_number)
	#print(columns_name[numb])
	mean_1=df[columns_name[numb]].mean()
	#print("columns_name and mean",mean_1)
	lst = [df[columns_name[numb]].mean()] * len(df.index)     
	## label data
	label = df[str(columns_name[numb])]
	## AI predicted data
	pred = lst
	# MAE計算
	mae = mean_absolute_error(label, pred)
	#print('MAE : {:.3f}'.format(mae))
	# {:.3f}で小数点以下は3桁で表示
	# RMSE計算
	rmse = np.sqrt(mean_squared_error(label, pred))
	#print('RMSE : {:.3f}'.format(rmse))
	index1 = ["mean", 'MAE_mean', 'RMSE_mean']
	columns1 =[str(columns_name[numb])]
	Calculation=pd.DataFrame(data=[df[columns_name[numb]].mean(),format(mae),format(rmse)], index=index1, columns=columns1)
	#print(Calculation)
	return Calculation
