import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("/content/dataAnalytics_LAB03.csv")

x = dataset['Body Weight Index'].values.reshape(-1, 1)
y = dataset['Blood Pressure(mmHg)'].values.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y)

w_0 = regressor.intercept_
w_1 = regressor.coef_
print('Interception : ', w_0)
print('Coefficient : ', w_1)

score = regressor.score(x, y)
print('Score: ', score)
print('Accuracy: ' + str(score * 100) + '%')

y_pred = regressor.predict(x)
print('Predict : ', y_pred)

plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Body Weight Index vs Blood Pressure(mmHg)')
plt.xlabel("Body Weight Index")
plt.ylabel("Blood Pressure(mmHg)")
plt.show()