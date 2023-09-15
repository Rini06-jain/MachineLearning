""" IMPORTING LIBRARIES """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge , Lasso,RidgeCV , LassoCV, ElasticNet , ElasticNetCV , LinearRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:\\Users\\Abhishek\\Downloads\\admission_data.csv")
#print(df.head())
#print(df.describe())
#print(df.isnull().sum())df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
#sns.histplot(df)
#sns.barplot(df)
#plt.boxplot(df)
#plt.show()
x=df.drop(columns=[ 'Chance of Admit '])
y=df[[ 'Chance of Admit ']]
#print(x,y)
#NORMALIATION/STANDARIZATION
#---------------
scaler=StandardScaler()
arr=scaler.fit_transform(x)
df1=pd.DataFrame(arr)
#print(df1)
#print(df1.describe())

#MULTICOLINEARITY
#-------------------
'''from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_df = pd.DataFrame()
vif_df['vif']=[variance_inflation_factor(arr,i)for i in range(arr.shape[1])]
vif_df["Features"]=x.columns
print(vif_df)'''

x_train ,x_test ,y_train , y_test=train_test_split(arr,y,test_size=0.25 ,random_state=365)
lr=LinearRegression()
lr.fit(x_train , y_train)
test1=scaler.transform([[337 ,  118  ,  4  ,4.5  , 4.5 , 9.65   , 1] ])
pred=lr.predict(test1)
pred2=lr.predict(x_test)

#print(pred)
'''s= lr.score(x_test,y_test)
print(s)'''
Lassocv= LassoCV(alphas=None ,cv=8, max_iter=2000)
Lassocv.fit(x_train , y_train)
lasso= Lasso(alpha=Lassocv.alpha_)
lasso.fit(x_train , y_train)
cv=lasso.score(x_train , y_train)
#print(cv)
elast=ElasticNetCV(alphas=None , cv=5)
elast.fit(x_train , y_train)
E=ElasticNet(alpha=elast.alpha_ , l1_ratio=elast.l1_ratio_)
E.fit(x_train , y_train)
cv2=E.score(x_train , y_train)
#print(cv2)




