import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv("Datasets/prostate.csv")
#df.drop('svi',axis=1,inplace=True)
x=df.iloc[:,0:7]
y=df.iloc[:,8]
x=x.values
y=y.values
x=np.matrix(x)
y=np.matrix(y)


y=np.reshape(y,(y.shape[1],1))

s=0.0

for i in range(1,100):
	x_train,x_test,y_train,y_test=train_test_split(x,y)

	x_train=(x_train-np.mean(x_train,axis=0))/np.std(x_train,axis=0)
	x_test=(x_test-np.mean(x_test,axis=0))/np.std(x_test,axis=0)

#y=(y-np.mean(y))/np.std(y);

	x_train=np.insert(x_train,0,1,axis=1)
	x_test=np.insert(x_test,0,1,axis=1)


	w=np.linalg.inv((np.transpose(x_train)*x_train))*np.transpose(x_train)*(y_train)

	y_hat=x_test*w;

	s=s+np.mean(np.power(y_hat-y_test,2))

print(s/100)

