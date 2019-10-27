import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
diamonds=pd.read_csv('diamonds.csv')
diamonds=diamonds.drop(["Unnamed: 0"],axis=1)
#FINDING HOW MANY MISSING VALUES ARE THERE 
def find_missing_values(diamonds):
    num_rows_x=(diamonds.x==0).sum()
    num_rows_y=(diamonds.y==0).sum()
    num_rows_z=(diamonds.z==0).sum()
    s=(diamonds.depth==0).sum()
    print(num_rows_x,num_rows_y,num_rows_z,s)

#REPLACE 0's WITH NAN
find_missing_values(diamonds)
diamonds[["x","y","z"]]=diamonds[["x","y","z"]].replace(0,np.NaN)
#find_missing_values and drop them(diamonds)
diamonds.dropna(inplace=True)
print(diamonds.shape)
#ONE HOT ENCODING
new_cleaned_df=pd.get_dummies(diamonds)
col=new_cleaned_df.columns
diamonds_cleaned=pd.DataFrame(new_cleaned_df,columns=col)

#FEATURE SCALING
X=StandardScaler()
a=pd.DataFrame(X.fit_transform(diamonds_cleaned[['carat','depth','x','y','z','table']]),columns=['carat','dept','x','y','z','table'],index=diamonds_cleaned.index)
new_processed_diamond=diamonds_cleaned.copy(deep=True)
new_processed_diamond[['carat','depth','x','y','z','table']]=a[['carat','dept','x','y','z','table']]
print(new_processed_diamond.head())

#PREPARE TEST AND TRAINING SET
train_data=new_processed_diamond.drop(["price"],axis=1)
test_data=new_processed_diamond["price"]

x_train,x_test,y_train,y_test=train_test_split(train_data,test_data,test_size=0.2,random_state=42)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

#CHOOSE THE RIGHT MODEL
model=RandomForestRegressor()
model.fit(x_train,y_train)



result=model.predict(x_test)
print(result[-5:])
print(y_test[-5:])

print("Accuracy  :",model.score(x_test,y_test))










