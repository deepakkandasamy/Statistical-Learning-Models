import numpy as np
import pandas as pd

df=pd.read_csv("Datasets/Boston.csv",index_col=0)
y = df.iloc[:,  13].values
df = (df - df.mean())/df.std() 
X = df.iloc[:, :13].values

centered_y=y-np.mean(y);

h=2 #bandwidth

iterations=10;
for iterator in xrange(iterations):
	for i in xrange(13):
		
