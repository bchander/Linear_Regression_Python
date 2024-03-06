'''......Importing Libraries......'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

'''.....Importing data as pandas dataframe......'''
input_data = pd.read_csv('data.txt', header = None);

'''.....Splitting data into Inputs/Target.......'''
x_value = input_data[[0]]
y_value = input_data[[1]]
m = len(x_value)

'''.....Visualize the input data.......'''
f = mplt.figure(1)
mplt.rcParams['figure.dpi'] = 300
mplt.scatter(x_value, y_value, color = 'red')
mplt.title("Input Data")
mplt.xlabel('X data')
mplt.ylabel('Y data')
f.show()

'''.....Creating Train/Test data.......'''
x_train, x_test, y_train, y_test = train_test_split(x_value,
                                                    y_value, 
                                                    shuffle=True,
                                                    test_size=0.3)
#x_train = x_value[:round(m/2)]
#y_train = y_value[:round(m/2)]
# selecting rest half as test data
#x_test = x_value[round(m/2):]
#y_test = y_value[round(m/2):]

'''..................................
..........Linear Regression..........
.....................................'''

# setting regression object
model = linear_model.LinearRegression()

#Training the model using train data (x_trin, y_train)
model.fit(x_train, y_train)

# Making predictions on unseen data (x_test)
y_predict = model.predict(x_test)


# checking the error (difference) predicted values and respective actual values
error = y_predict - y_test

#Measuring model performance using RMSE (Root Mean Square Errors)
RMSE = np.sqrt(np.mean(np.square(y_predict - y_test)))
variance = model.score(x_test, y_test)
# find necessary coefficents - correlation coeff, RMSE, Variance
print("\n Coefficinets  = \n", model.coef_)
print("RMSE: %0.2f" % RMSE)
print("Variance: %0.2f" % variance)

# Plotting the regression line/curve
g = mplt.figure(2)

mplt.rcParams['font.size'] = '12'
mplt.rcParams['figure.dpi'] = 600

mplt.scatter(x_test, y_test, color = 'black')
mplt.plot(x_test, y_predict, color = 'blue', linewidth = 2)
mplt.title("Simple Regression")
#mplt.xticks(())
#mplt.yticks(())
mplt.xlabel('X')
mplt.ylabel('Y')
g.show()


'''...............................................
...............Polynomial Regression..............
..................................................'''

from sklearn.preprocessing import PolynomialFeatures

# Creating polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(x_value)

x_train, x_test, y_train, y_test = train_test_split(poly_features,
                                                    y_value, 
                                                    shuffle=True,
                                                    test_size=0.3)
# Defining model based on the polynomial features
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# checking the error (difference) predicted values and respective actual values
error_poly = y_pred - y_test

#Measuring model performance using RMSE (Root Mean Square Errors)
RMSE_poly = np.sqrt(np.mean(np.square(y_pred - y_test)))
#ariance_poly = model.score(x_test, y_test)
# find necessary coefficents - correlation coeff, RMSE, Variance
print("\n Coefficinets  = \n", model.coef_)
print("RMSE: %0.2f" % RMSE)
#print("Variance: %0.2f" % variance)


poly_data = pd.DataFrame(zip(x_test[:, 0], y_pred[:,0]), columns=['x', 'y'])
sorted_data = poly_data.sort_values(by = 'x')

# Plotting the regression line/curve
g = mplt.figure(2)
mplt.rcParams['font.size'] = '12'
mplt.rcParams['figure.dpi'] = 600
mplt.scatter(x_test[:, 0],  y_test, color = 'black')
mplt.plot(sorted_data['x'],  sorted_data['y'], color = 'blue', linewidth = 2, label = 'Poly Regression')
mplt.title("Polynomial Regression degree 3")
mplt.xlabel('X')
mplt.ylabel('Y')
g.show()
