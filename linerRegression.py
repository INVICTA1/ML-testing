import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math 
from IPython.display import display
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

path=r"data\oil_exxon.xlsx"
price_data = pd.read_excel(path)
print(price_data.head())
    # установить индекс, равный столбцу даты, а затем удалить старый столбец даты

price_data.index = pd.to_datetime(price_data['date'])#преобразует объект в дату и время
price_data = price_data.drop(['date'], axis = 1)#удаляем индексированный столбец
print('first five lines of data \n',price_data.head())

#rename column
print("check the data types:\n",price_data.dtypes)
new_column_names = {'exon_price':'exxon_price'}
price_data = price_data.rename(columns = new_column_names)
print('\n head auto renaming \n',price_data.head(),'\n')

print('пропущенные занчения \n',price_data.isna().any(),'\n')#проверяем есть ли пропущенные значения (true-есть, false-нет)
#isna()-проверяет каждый элемент , any -возвращает False, если хотя бы один элемент в серии или вдоль оси Dataframe не является True или эквивалентным
price_data = price_data.dropna()# удаляем все отсутствующие значения(параметры 0(по умолчанию) удаляет строки, 1-столбцы)
print(price_data.isna().any())
x = price_data['exxon_price']
y = price_data['oil_price']
# создаем диаграмму рассеяния.

plt.plot(x, y, 'o', color ='cadetblue', label = 'Daily Price')
#форматирование графика
plt.title("Exxon Vs. Oil")
plt.xlabel("Exxon Mobile")
plt.ylabel("Oil")
plt.legend()#нанести на оси данные
plt.show()

# измерим  корреляцию
price_data.corr()

price_data.describe()# статистическая сводка для описания набора данных
#создаем гистограмму для каждого столбца данных
price_data.hist(grid = False, color = 'cadetblue')
plt.show()
exxon_kurtosis = kurtosis(price_data['exxon_price'], fisher = True)
oil_kurtosis = kurtosis(price_data['oil_price'], fisher = True)

# вычисляем асимметрию
exxon_skew = skew(price_data['exxon_price'])
oil_skew = skew(price_data['oil_price'])

display("Exxon Excess Kurtosis: {:.2}".format(exxon_kurtosis))
display("Oil Excess Kurtosis: {:.2}".format(oil_kurtosis))
#вычисляем асимметрию
display("Exxon Skew: {:.2}".format(exxon_skew))
display("Oil Skew: {:.2}".format(oil_skew))
display('Exxon')
display(stats.kurtosistest(price_data['exxon_price']))
display('Oil')
display(stats.kurtosistest(price_data['oil_price']))
# выполнить тест перекоса

display('Exxon')
display(stats.skewtest(price_data['exxon_price']))
display('Oil')
display(stats.skewtest(price_data['oil_price']))
# определить нашу входную переменную (X) и выходную переменную.

Y = price_data.drop('oil_price', axis = 1)
X = price_data[['oil_price']]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
# создать объект модели линейной регрессии
regression_model = LinearRegression()

# пройти через набор данных X_train & y_train.
regression_model.fit(X_train, y_train)
intercept = regression_model.intercept_[0]
coefficient = regression_model.coef_[0][0]

print("The Coefficient for our model is {:.2}".format(coefficient))
print("The intercept for our model is {:.4}".format(intercept))
prediction = regression_model.predict([[67.33]])
predicted_value = prediction[0][0]
print("The predicted value is {:.4}".format(predicted_value))
y_predict = regression_model.predict(X_test)

# Show the first 5 predictions
y_predict[:5]
X2 = sm.add_constant(X)

# создать  a OLS модель.
model = sm.OLS(Y, X2)

# fit the data
est = model.fit()
est.conf_int()

est.pvalues
# вычислить среднеквадратичную ошибку.
model_mse = mean_squared_error(y_test, y_predict)

# вычислить среднюю абсолютную ошибку.
model_mae = mean_absolute_error(y_test, y_predict)

# вычислить среднеквадратичную ошибку
model_rmse =  math.sqrt(model_mse)

# display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))
model_r2 = r2_score(y_test, y_predict)
print("R2: {:.2}".format(model_r2))
print(est.summary())
(y_test - y_predict).hist(grid = False, color = 'royalblue')
plt.title("Model Residuals")
plt.show()
plt.scatter(X_test, y_test,  color='gainsboro', label = 'Price')
plt.plot(X_test, y_predict, color='royalblue', linewidth = 3, linestyle= '-',label ='Regression Line')

plt.title("Linear Regression Exxon Mobile Vs. Oil")
plt.xlabel("Oil")
plt.ylabel("Exxon Mobile")
plt.legend()
plt.show()

# Коэффициенты
print('Oil coefficient:' + '\033[1m' + '{:.2}''\033[0m'.format(regression_model.coef_[0][0]))

# Среднеквадратичная ошибка
print('Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(model_mse))

# Среднеквадратичная ошибка
print('Root Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(math.sqrt(model_mse)))

# Объясненная оценка дисперсии: 1 - идеальное предсказание
print('R2 score: '+ '\033[1m' + '{:.2}''\033[0m'.format(r2_score(y_test,y_predict)))

with open('my_linear_regression.sav', 'wb') as f:
    pickle.dump(regression_model, f)

# load it back in.
with open('my_linear_regression.sav', 'rb') as pickle_file:
    regression_model_2 = pickle.load(pickle_file)

# сделаем новый прогноз.
regression_model_2.predict([[67.33]])