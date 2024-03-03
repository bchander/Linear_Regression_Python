#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt
from sklearn import datasets, linear_model

# Taking Dataset using Pandas
input_data = pd.read_csv('data.txt', header = None);
# the data doesn't have any headerm, if the data has header say 'x, 'y', 
# then use input_data[['x']]
# if u us semicolon and hot enter, thenthe cursor will go the next line inline with the semicolon

x_value = input_data[[0]]
y_value = input_data[[1]]
m = len(x_value)
# visualize the plot as scatter
f = mplt.figure(1)
mplt.rcParams['figure.dpi'] = 600
mplt.scatter(x_value, y_value, color = 'red')
mplt.title("Input Data")
mplt.xlabel('X data')
mplt.ylabel('Y data')
f.show()
# selecting half the data as train
x_train = x_value[:round(m/2)]
y_train = y_value[:round(m/2)]

# selecting rest half as test data
x_test = x_value[round(m/2):]
y_test = y_value[round(m/2):]

# setting regression object
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
#Root Mean sqare error scores
error = y_predict - y_test
RMSE = np.sqrt(np.mean(np.square(y_predict - y_test)))
variance = reg.score(x_test, y_test)
# find necessary coefficents - correlation coeff, RMSE, Variance
print("\n Coefficinets  = \n", reg.coef_)
print("RMSE: %0.2f" % RMSE)
print("Variance: %0.2f" % variance)

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

# raw input allows the prog to wait until enter is hit 

#mplt.scatter(x_value, y_value)


'''
SOME IMPORTANT NOTES 

fit takes argumemts : fit(X (training data),y_predict(target data), sampe_weight = None)
score(X, y_predict, sample_weight = None)

'''
