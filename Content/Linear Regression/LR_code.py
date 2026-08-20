import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_california_housing
dataset=fetch_california_housing()

x=pd.DataFrame(dataset.data) #This will contain the independent variable.
y=dataset.target    #This will contain the dependent variable.

x.columns = dataset.feature_names

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.30,random_state=42
)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

from sklearn.model_selection import cross_val_score
mse=cross_val_score(model,x_train,y_train,scoring='neg_mean_squared_error',cv=5)
np.mean(mse)

reg_pred=model.predict(x_test)

from sklearn.metrics import r2_score
score=r2_score(reg_pred,y_test)
print(score)