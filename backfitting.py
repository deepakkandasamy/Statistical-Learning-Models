import numpy as np
import pandas as pd
from Kernels.epanechnikov import epanechnikov


df=pd.read_csv("Datasets/Boston.csv",index_col=0)
y = df.iloc[:,  13].values
df = (df - df.mean())/df.std() 
X = df.iloc[:, :13].values

num_attributes=len(X[0])
num_records=len(X);
centered_y=y-np.mean(y)

h=2 #bandwidth

y_hat=[0]*num_records;
const=[0]*num_attributes;
	
iterations=10;
for iterator in xrange(iterations):
	for j in xrange(num_attributes):
		for i in xrange(num_records):
			sum_k=0
			for k in xrange(num_attributes):
				denom=0
				for m in xrange(num_records):
					denom=denom+(epanechnikov((X[i,k]-X[m,k])/h)-const[k])
				for m in xrange(num_records):
					sum_k=sum_k+y[m]*(epanechnikov((X[i,k]-X[m,k])/h)/denom-const[k])
			y_hat[i]=centered_y[i]-sum_k
		sum_i=0
		denom=0
		for m in xrange(num_records):
			denom=denom+(epanechnikov((X[i,j]-X[m,j])/h)-const[j])
		for m in xrange(num_records):
			sum_i=sum_i+y_hat[m]*(epanechnikov((X[i,j]-X[m,j])/h)-const[j])
		const[j]=(1.0/num_attributes)*sum_i;
	print(const)
