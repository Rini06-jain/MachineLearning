import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge , Lasso,RidgeCV , LassoCV, ElasticNet , ElasticNetCV , LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix ,roc_curve, roc_auc_score
df=pd.read_csv("C:\\Users\\Abhishek\\Downloads\\diabetes.csv")
#print(df.head())
#print(df.describe())
#print(df.isnull().sum())


"""REPLACING ZEROS"""  

df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['SkinThinckness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())
df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
#sns.histplot(df)


"""FINDING AND REMOVING OUTLIERS"""

'''plt.xticks(rotation='vertical')
sns.boxplot(df)
plt.show()'''

q=df['BMI'].quantile(.95)
df_new=df[df['BMI']<q]

q=df['Glucose'].quantile(.95)
df_new=df[df['Glucose']<q]
q=df['Age'].quantile(.92)
df_new=df[df['Age']<q]
q=df['SkinThickness'].quantile(.95)
df_new=df[df['SkinThickness']<q]
q=df['Insulin'].quantile(.70)
df_new=df[df['Insulin']<q]
q=df['BloodPressure'].quantile(.95)
df_new=df[df['BloodPressure']<q]
#print(df_new)

q1,q3=np.percentile(df['Insulin'],[25,75])
iqr= q3-q1
UL=q3+(1.5*iqr)
LL= q1-(1.5*iqr)
#print(UL ,LL)
df_new = df[df['Insulin']<LL]
df_new = df[df['Insulin']>UL]
'''plt.xticks(rotation='vertical')
sns.boxplot(df_new)
plt.show()'''

"""MULTICOLINEARITY"""
x=df_new.drop(columns=[ 'Outcome'])
y=df_new[[ 'Outcome']]
scaler=StandardScaler()
arr=scaler.fit_transform(x)
df1=pd.DataFrame(arr)
'''from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_df = pd.DataFrame()
vif_df['vif']=[variance_inflation_factor(arr,i)for i in range(arr.shape[1])]
vif_df["Features"]=df.columns'''
#print(vif_df)
x_train ,x_test ,y_train , y_test=train_test_split(arr,y,test_size=0.25 ,random_state=365)
#print(x_train , y_train)
lor = LogisticRegression()
lor.fit(x_train,y_train)
test1=scaler.transform([[  0 , 137.0,40.0 ,35 ,168.000000 , 43.1 , 2.288 , 33,35.00  ]])
y_pred=lor.predict(x_test)
pred=lor.predict(test1)
#print(pred)
tn ,fp ,fn ,tp=confusion_matrix(y_test ,y_pred).ravel()
acc = (tp-tn)/(tp+tn+fp+fn)
prec = tp/(tp+fp)
recall=tp/(tp+fn)

print(acc,prec,recall)
h=roc_curve(y_test ,y_pred)
#print(h)





