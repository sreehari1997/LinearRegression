import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split  --depreciated
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('USA_Housing.csv')

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)

lm = LinearRegression()
#trainig
lm.fit(X_train, y_train,)

predictions = lm.predict(X_test)
#visualisation
#plt.scatter(y_test, predictions)
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

print("MAE : {} \nMSE : {} \nRMSE : {} ".format(mae, mse, rmse))
sns.distplot(y_test - predictions)
plt.show()





