import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data=pd.read_csv("Mall_Customers.csv")
data.isnull().sum()
data=data.drop(columns=["CustomerID"])
data["Genre"]=data["Genre"].map({"Male":0, "Female":1})
#print(data.dtypes)
x=data[["Genre","Age","Annual Income (k$)"]]
y=data["Spending Score (1-100)"]
"""print(data.describe())
print(data.groupby("Genre")["Spending Score (1-100)"].mean())
print(data[["Age", "Spending Score (1-100)"]].corr())
print(data[["Annual Income (k$)", "Spending Score (1-100)"]].corr())
print(data)"""
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=50)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


